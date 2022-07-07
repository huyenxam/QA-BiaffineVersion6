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
        self.bilstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=args.hidden_dim // 2,
                              num_layers=2, bidirectional=True, batch_first=True)
        self.feedStart = FeedforwardLayer(d_in=args.hidden_dim, d_hid=args.hidden_dim_ffw)
        self.feedEnd = FeedforwardLayer(d_in=args.hidden_dim, d_hid=args.hidden_dim_ffw)
        self.biaffine = BiaffineLayer(inSize1=args.hidden_dim, inSize2=args.hidden_dim, classSize=self.num_labels)


    def forward(self, input_ids_ctx=None, char_ids_ctx=None,  firstSWindices_ctx=None, attention_mask_ctx=None, 
    input_ids_ques=None, char_ids_ques=None,  firstSWindices_ques=None, attention_mask_ques=None):

        # x = [bs, max_sep, 768 + char_hidden_dim*2]
        ctx_word_embed = self.word_rep(input_ids=input_ids_ctx, attention_mask=attention_mask_ctx,
                                      first_subword=firstSWindices_ctx,
                                      char_ids=char_ids_ctx)

        ctx_len = ctx_word_embed.shape[1]
        # ctx_word_embed = [bs, ctx_len, 768 + char_hidden_dim*2]

        ques_word_embed = self.word_rep(input_ids=input_ids_ques, attention_mask=attention_mask_ques,
                                      first_subword=firstSWindices_ques,
                                      char_ids=char_ids_ques)
        # ques_word_embed = [bs, ques_len, 768 + char_hidden_dim*2]
        # emb_dim = 768 + char_hidden_dim 
        ques_len = ques_word_embed.shape[1]
        ## CREATE SIMILARITY MATRIX
        ctx_ = ctx_word_embed.unsqueeze(2).repeat(1,1,ques_word_embed,1)
        # [bs, ctx_len, 1, emb_dim*2] => [bs, ctx_len, ques_len, emb_dim*2]
        
        ques_ = ques_word_embed.unsqueeze(2).repeat(1,1,ctx_word_embed,1)
        # [bs, ques_len, 1, emb_dim*2] => [bs, ctx_len, ques_len, emb_dim*2]
        
        elementwise_prod = torch.mul(ctx_, ques_)
        # [bs, ctx_len, ques_len, emb_dim*2]

        alpha = torch.cat([ctx_, ques_, elementwise_prod], dim=3)
        # [bs, ctx_len, ques_len, emb_dim*6]
        
        similarity_matrix = self.similarity_weight(alpha).view(-1, ctx_len, ques_len)
        # [bs, ctx_len, ques_len]

        ## CALCULATE CONTEXT2QUERY ATTENTION
        
        a = F.softmax(similarity_matrix, dim=-1)
        # [bs, ctx_len, ques_len]
        
        c2q = torch.bmm(a, ques_word_embed)
        # [bs] ([ctx_len, ques_len] X [ques_len, emb_dim*2]) => [bs, ctx_len, emb_dim*2]
        
        
        ## CALCULATE QUERY2CONTEXT ATTENTION
        
        b = F.softmax(torch.max(similarity_matrix,2)[0], dim=-1)
        # [bs, ctx_len]
        
        b = b.unsqueeze(1)
        # [bs, 1, ctx_len]
        
        q2c = torch.bmm(b, ctx_word_embed)
        # [bs] ([bs, 1, ctx_len] X [bs, ctx_len, emb_dim*2]) => [bs, 1, emb_dim*2]
        
        q2c = q2c.repeat(1, ctx_len, 1)
        # [bs, ctx_len, emb_dim*2]
        
        ## QUERY AWARE REPRESENTATION
        
        x = torch.cat([ctx_word_embed, c2q, 
                       torch.mul(ctx_word_embed,c2q), 
                       torch.mul(ctx_word_embed, q2c)], dim=2)
        
        # [bs, ctx_len, emb_dim*8]

        x, _ = self.bilstm(x)
        # x = [bs, ctx_len, hidden_dim]
        start = self.feedStart(x)
        # start = [bs, ctx_len, hidden_dim]
        end = self.feedEnd(x)
        # end = [bs, ctx_len, hidden_dim]
        score = self.biaffine(start, end)
        # score = [bs, ctx_len, ctx_len, 2]
        return score