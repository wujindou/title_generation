class BARTwithDetection(BaseModel):
    def __init__(self):
        super().__init__()
        self.bart = build_transformer_model(config_path, checkpoint_path, model='bart', keep_tokens=keep_tokens, segment_vocab_size=0)
        self.linear = nn.Linear(768,1)
    def forward(self,inputs):
        output =  self.bart(inputs)
        last_hidden, _, y_pred = output
        logits = torch.squeeze(self.linear(last_hidden),2)
        return (_,logits,y_pred)
    def get_bart(self):
        return self.bart
    
class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss2 = nn.CrossEntropyLoss(ignore_index=0)
        self.ner_loss = nn.BCEWithLogitsLoss()
        self.linear = nn.Linear(768,1)
    def forward(self,outputs,labels):
        _,logits,y_pred = outputs
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        loss2 = self.loss2(y_pred,labels[0])
        loss1 = self.ner_loss(logits,labels[1])
        return {'loss':loss2+loss1,'loss1':loss1,'loss2':loss2}
class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, outputs, y_true):
        _, _, y_pred = outputs
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        return super().forward(y_pred, y_true)
model = BARTwithDetection().to(device)
model.compile(loss=Loss(), optimizer=optim.Adam(model.parameters(), 1.5e-5))
