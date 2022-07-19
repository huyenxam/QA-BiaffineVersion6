from torch import nn
from model.layer import WordRep, FeedforwardLayer, BiaffineLayer
from transformers import AutoConfig
import torch
import torch.nn.functional as F

class BiaffineNER(nn.Module):
    def __init__(self, args):
        super(BiaffineNER, self).__init__()
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.num_labels = args.num_labels
        self.lstm_input_size = args.num_layer_bert * config.hidden_size
        if args.use_char:
            self.lstm_input_size = self.lstm_input_size + 2 * args.char_hidden_dim

        self.word_rep = WordRep(args)
        self.similarity_weight = nn.Linear(args.hidden_dim*3, 1, bias=False)
        # self.bilstm_seq = nn.LSTM(input_size=self.lstm_input_size, hidden_size=100 // 2,
        #                       num_layers=2, bidirectional=True, batch_first=True)
                              
        self.bilstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=args.hidden_dim // 2,
                              num_layers=2, bidirectional=True, batch_first=True)

        self.feedStart = FeedforwardLayer(d_in=args.hidden_dim*4, d_hid=args.hidden_dim_ffw)
        self.feedEnd = FeedforwardLayer(d_in=args.hidden_dim*4, d_hid=args.hidden_dim_ffw)
        self.biaffine = BiaffineLayer(inSize1=args.hidden_dim*4, inSize2=args.hidden_dim*4, classSize=self.num_labels)


    def forward(self, input_ids_ctx=None, char_ids_ctx=None,  firstSWindices_ctx=None, attention_mask_ctx=None, 
    input_ids_ques=None, char_ids_ques=None,  firstSWindices_ques=None, attention_mask_ques=None):

        # x = [bs, max_sep, 768 + char_hidden_dim*2]
        ctx_word_embed = self.word_rep(input_ids=input_ids_ctx, attention_mask=attention_mask_ctx,
                                      first_subword=firstSWindices_ctx,
                                      char_ids=char_ids_ctx)
        # ctx_word_embed = [bs, ctx_len, 768 + char_hidden_dim * 2]
        ctx_len = ctx_word_embed.shape[1]
        ctx_word_embed, _ = self.bilstm(ctx_word_embed)
        # ctx_word_embed = [bs, ctx_len, hidden_dim]

        ques_word_embed = self.word_rep(input_ids=input_ids_ques, attention_mask=attention_mask_ques,
                                      first_subword=firstSWindices_ques,
                                      char_ids=char_ids_ques)
        # ques_word_embed = [bs, ques_len, 768 + char_hidden_dim * 2]
        ques_len = ques_word_embed.shape[1]
        ques_word_embed, _ = self.bilstm(ques_word_embed)
        # ques_word_embed = [bs, ques_len, hidden_dim]

        ## CREATE SIMILARITY MATRIX
        ctx_ = ctx_word_embed.unsqueeze(2).repeat(1,1,ques_len,1)
        # [bs, ctx_len, 1, hidden_dim] => [bs, ctx_len, ques_len, hidden_dim]
        
        ques_ = ques_word_embed.unsqueeze(1).repeat(1,ctx_len,1,1)
        # [bs, 1, ques_len, hidden_dim] => [bs, ctx_len, ques_len, hidden_dim]
        
        elementwise_prod = torch.mul(ctx_, ques_)
        # [bs, ctx_len, ques_len, hidden_dim]

        alpha = torch.cat([ctx_, ques_, elementwise_prod], dim=3)
        # [bs, ctx_len, ques_len, hidden_dim*3]
        
        similarity_matrix = self.similarity_weight(alpha).view(-1, ctx_len, ques_len)
        # [bs, ctx_len, ques_len]

        ## CALCULATE CONTEXT2QUERY ATTENTION
        
        a = F.softmax(similarity_matrix, dim=-1)
        # [bs, ctx_len, ques_len]
        
        c2q = torch.bmm(a, ques_word_embed)
        # [bs] ([ctx_len, ques_len] X [ques_len, hidden_dim]) => [bs, ctx_len, hidden_dim]
        
        
        ## CALCULATE QUERY2CONTEXT ATTENTION
        
        b = F.softmax(torch.max(similarity_matrix,2)[0], dim=-1)
        # [bs, ctx_len]
        
        b = b.unsqueeze(1)
        # [bs, 1, ctx_len]
        
        q2c = torch.bmm(b, ctx_word_embed)
        # [bs] ([bs, 1, ctx_len] X [bs, ctx_len, hidden_dim]) => [bs, 1, hidden_dim]
        
        q2c = q2c.repeat(1, ctx_len, 1)
        # [bs, ctx_len, hidden_dim]
        
        ## QUERY AWARE REPRESENTATION
        
        x = torch.cat([ctx_word_embed, c2q, 
                       torch.mul(ctx_word_embed,c2q), 
                       torch.mul(ctx_word_embed, q2c)], dim=2)
        print(x.shape)
        # [bs, ctx_len, hidden_dim*4]

        # x, _ = self.bilstm(x)
        # # x = [bs, ctx_len, hidden_dim]
        start = self.feedStart(x)
        # start = [bs, ctx_len, hidden_dim*4]
        end = self.feedEnd(x)
        # end = [bs, ctx_len, hidden_dim*4]
        score = self.biaffine(start, end)
        # score = [bs, ctx_len, ctx_len, 2]
        return score