
import sys
import torch
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from dataset import ListopsDataset, CifarDataset
import argparse
import random
import glob
from transformers import AdamW, AutoModelForSequenceClassification , GPT2Tokenizer, GPT2Model,AutoModelForPreTraining, AutoTokenizer, AutoModelForSeq2SeqLM
from pytorch_lightning.callbacks import ModelCheckpoint


def compute_exact_match(predicted_answer, correct_answer):
    predicted_answer = predicted_answer.strip().lower()
    correct_answer = correct_answer.strip().lower()
    return predicted_answer[-1] == correct_answer[-1]


class T5model(pl.LightningModule):
    
    
    def __init__(self, hparams, train_dataloader, val_dataloader):
        super().__init__()
        
        self.hparams = hparams

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.hparams.model_name_or_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))



        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        


    def prepare_batch(self, sequence, label):




        input_dict = self.tokenizer.batch_encode_plus(
            list(sequence), padding=True, truncation=False, return_tensors='pt')


        labels = self.tokenizer.batch_encode_plus(
            list(label), padding=True, truncation=False, return_tensors='pt')['input_ids']
        

        input_ids = input_dict['input_ids'].to(self.model.device)
        attention_mask = input_dict['attention_mask'].to(self.model.device)
        labels = labels.to(self.model.device)

        return input_ids, attention_mask, labels        
    
    
    def training_step(self, batch, batch_nb):
        sequence, label = batch

        # Log every power of two.
        if batch_nb & (batch_nb - 1) == 0:
            print(sequence[0])
            print(label[0])

        input_ids, attention_mask, labels = self.prepare_batch(
            sequence=sequence, label=label)

        loss = self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels)[0]

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}    
    
    
    def inference_step(self, batch, batch_nb: int):
        sequence, label = batch

        input_ids, attention_mask, _ = self.prepare_batch(
            sequence=sequence, label=label)

        batch_outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_length=self.hparams.max_seq_length)

        predicted_answers = [
            self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for output in batch_outputs]

        exact_matches = [
            compute_exact_match(predicted_answer=predicted_answer, correct_answer=correct_answer)
            for predicted_answer, correct_answer in zip(predicted_answers, label)]

        # Log every power of two.
        if batch_nb & (batch_nb - 1) == 0:
            print('sequence:', sequence[0])
            print('Correct:  ', label[0])
            print('Predicted:', predicted_answers[0].encode('utf-8'))
            print('Exact?', exact_matches[0])

        metrics = {'exact_matches': exact_matches}
        return metrics    
    
    def validation_step(self, batch, batch_nb):
        return self.inference_step(batch, batch_nb)


    def validation_epoch_end(self, outputs):
        exact_matches = []
        for x in outputs:
            exact_matches.extend(x['exact_matches'])
        exact_match = sum(exact_matches) / len(exact_matches)

        metrics = {'val_exact_match': exact_match}

        output = metrics.copy()
        output['progress_bar'] = metrics

        return output    



    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader



    def get_optimizer(self):
        optimizer_name = self.hparams.optimizer
        scheduler_name = self.hparams.scheduler
        lr = self.hparams.lr
        weight_decay = self.hparams.weight_decay

        optimizer = getattr(torch.optim, optimizer_name)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        optimizer = optimizer(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)

        print(f'=> Using {optimizer_name} optimizer')

        if scheduler_name == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
            print(f'=> Using StepLR (step_size = {self.hparams.step_size}, gamma = {self.hparams.gamma})')
        else:
            raise Exception(f'Scheduler not implemented: {scheduler_name}')

        return [optimizer], [scheduler]

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evalute T5.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save checkpoint and results.')
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument("--seed", default=123, type=int, help="Seed.")
    parser.add_argument('--max_seq_length', type=int, default=512, help='Maximum sequence length (in tokens).')
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--val_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument("--lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--scheduler', type=str, default='StepLR',
                        help='learning rate scheduler. Currently, only StepLR is supported.)')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma factor for ExponentialLR or StepLR')
    parser.add_argument('--step_size', type=int, default=2, help='period of learning rate decay (StepLR)')
    parser.add_argument('--t_0', type=int, default=2,
                        help='number of iterations for the first restart (CosineAnnealingWarmRestarts)')
    parser.add_argument('--t_mult', type=int, default=2,
                        help='a factor increases t_i after a restart (CosineAnnealingWarmRestarts)')
    parser.add_argument("--num_workers", default=4, type=int, help="Number of CPU workers for loading data.")

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    print('args', args)

    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    pl.seed_everything(args.seed)

    dataset_train = ListopsDataset('basic_train.tsv')


    dataset_val =  ListopsDataset('basic_val.tsv')


    train_dataloader = DataLoader(dataset_train, batch_size=args.train_batch_size,
                                  shuffle=True, num_workers=args.num_workers)

    val_dataloader = DataLoader(dataset_val, batch_size=args.val_batch_size, shuffle=False,
                                num_workers=args.num_workers)


    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.output_dir, '{epoch}-{val_exact_match:.4f}'),
        verbose=False, save_last=False, save_top_k=1, mode='max', monitor='val_exact_match',
        save_weights_only=False, period=args.check_val_every_n_epoch)

    trainer = pl.Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback)

    model = T5model(hparams=args,
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader)

    trainer.fit(model)

    checkpoint_path = glob.glob(os.path.join(args.output_dir, '*.bin'))[0]
    model = T5model.load_from_checkpoint(checkpoint_path,
                                             train_dataloader=train_dataloader,
                                             val_dataloader=val_dataloader)

    results = trainer.test(model)

    output = {'seed': args.seed,
              'test_exact_match': results[0]['test_exact_match']}


    