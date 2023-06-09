from transformers import AutoModelForMultipleChoice, TrainingArguments
from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, T5ForConditionalGeneration
from transformers.file_utils import ModelOutput
import torch
from .model_transformer import MyTransformer, SemanticMatch
from torch import nn


def get_model(model_name):
    model = AutoModelForMultipleChoice.from_pretrained(model_name)
    return model


class CustomModel(nn.Module):
    def __init__(self, device_str):
        super(CustomModel, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        device = torch.device(device_str)
        self.encoder = BertGenerationEncoder.from_pretrained("bert-base-uncased")
        # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
        self.decoder = BertGenerationDecoder.from_pretrained(
            "bert-base-uncased", add_cross_attention=True, is_decoder=True
        )
        self.bert_generation = EncoderDecoderModel(encoder=self.encoder, decoder=self.decoder)
        self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')
        print(self.bert_generation)
        print(self.t5_model)

    def forward(self, q_ids, q_mask, qo_ids, qo_mask, clue_ids=None, answers=None):
        if answers is not None and clue_ids is not None:
            opt_score, output_sequences = self.get_option_score(q_ids, q_mask, qo_ids, qo_mask)
            local_device = self.bert_generation.device
            bert_output = self.bert_generation(input_ids=q_ids.to(local_device), attention_mask=q_mask.to(local_device),
                                      labels=clue_ids.to(local_device), decoder_input_ids=clue_ids.to(local_device))
            loss_ans = bert_output.loss
            loss = self.criterion(opt_score, answers)
            return self.alpha * loss + self.beta * loss_ans
        else:
            opt_score, output_sequences = self.get_option_score(q_ids, q_mask, qo_ids, qo_mask)
            return opt_score, output_sequences

    def get_option_score(self, q_ids, q_mask, qo_ids, qo_mask):
        local_device = self.t5_model.encoder.device
        t5_output = self.t5_model.encoder(input_ids=qo_ids.to(local_device), attention_mask=qo_mask.to(local_device))
        encoder_qo = t5_output[0]
        print(type(encoder_qo), encoder_qo.shape)
        t5_output = self.t5_model.encoder(input_ids=q_ids.to(local_device), attention_mask=q_mask.to(local_device))
        encoder_q = t5_output[0]
        print(type(encoder_q), encoder_q.shape)
        local_device = self.t5_model.device
        t5_output = self.t5_model.generate(
            encoder_outputs=ModelOutput(last_hidden_state=encoder_q.to(local_device)),
            attention_mask=q_mask.to(local_device),
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        print(type(t5_output), t5_output.shape)

        local_device = self.bert_generation.encoder.device
        bert_output = self.bert_generation.encoder(input_ids=qo_ids.to(local_device), attention_mask=qo_mask.to(local_device))
        encoder_qo = bert_output[0]
        print(type(encoder_qo), encoder_qo.shape)
        bert_output = self.bert_generation.encoder(input_ids=q_ids.to(local_device), attention_mask=q_mask.to(local_device))
        encoder_q = bert_output[0]

        local_device = self.bert_generation.device
        bert_output = self.bert_generation.decoder(
            inputs_embeds=encoder_q.to(local_device),
            attention_mask=q_mask.to(local_device),
            output_hidden_states=True
        )
        output_sequences = bert_output.logits
        output_sequences = output_sequences[:, 1:].contiguous()
        decoder_o = bert_output.hidden_states
        decoder_o = [item[-1] for item in decoder_o]
        decoder_o = torch.cat(decoder_o, dim=1)

        output_sequences_mask1 = output_sequences != 0
        output_sequences_mask2 = output_sequences != 1
        output_sequences_mask = output_sequences_mask1 * output_sequences_mask2
        output_sequences_mask = output_sequences_mask.long()
        decoder_qo = torch.cat([encoder_q, decoder_o], dim=1)
        output_sequences_mask = torch.cat([q_mask, output_sequences_mask], dim=1)
        local_device = self.transformer_laryer_de.device
        decoder_qo, _ = self.transformer_laryer_de(decoder_qo.to(local_device), output_sequences_mask.to(local_device))
        output_sequences_mask_ex = output_sequences_mask.unsqueeze(dim=1)
        output_sequences_mask_ex = output_sequences_mask_ex.expand(
            [output_sequences_mask_ex.size(0), self.choice_num, output_sequences_mask_ex.size(-1)]).contiguous()
        output_sequences_mask_ex = output_sequences_mask_ex.view(-1, output_sequences_mask.size(-1))
        decoder_qo = decoder_qo.unsqueeze(dim=1)
        decoder_qo = decoder_qo.expand(
            [decoder_qo.size(0), self.choice_num, decoder_qo.size(-2), decoder_qo.size(-1)]).contiguous()
        decoder_qo = decoder_qo.view(-1, decoder_qo.size(-2), decoder_qo.size(-1))
        local_device = self.semantic_matching.device
        semantic_vec, _, _ = self.semantic_matching(encoder_qo.to(local_device), decoder_qo.to(local_device),
                                                    qo_mask.to(local_device), output_sequences_mask_ex.to(local_device))
        local_device = self.option_linear.device
        opt_score = self.option_linear(semantic_vec.to(local_device)).view(-1, self.choice_num)

        return opt_score, output_sequences


def get_model_aug(device):
    return CustomModel(device)
