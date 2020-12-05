# Kaggle Mechanism Of Action(MOA) 71st/4373 place Solution
Thanks for the organizers for this long-waited tabular competition; it was both fun and challenging. To be honest, I am not good at tabular competitions at all. Thanks to all the public notebook contributors, I learned a lot from all of you guys; you guys are the true heroes of this competition. I also want to say sorry to my friend and ex-teammate @gogo827jz for missing his best submission; without your sharing, this competition wouldn't be as popular. I also want to say thank you to my teammates @wangyijia @nurmannaz for the contributions. Additionally, congrats to my previous teammate @tiandaye for winning his first gold medal🏅️ and my teammate @nurmannaz for getting his first kaggle medal🥈.

## TL;DR
Our final solution is mainly based on public kernels in terms of features. I have no idea how to do feature engineering and I don't know how to validate them. Tried couple methods, they all deteriorate the CV except PCA. With my model architectures and the abuse of pseudo labels, we were able to train some solid models. Along with the public kernel: https://www.kaggle.com/kushal1506/moa-pretrained-non-scored-targets-as-meta-features by @kushal1506, we were able to achieve 0.01815 public leaderboard and 0.01610 private leaderboard.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5168115%2Ff19ece2ff6e5ad0f0895cf56f3e4b91f%2Fmoa_final_solution.png?generation=1606786588701522&alt=media)

## Timeline
I started this competition at the beginning and tried a few submissions. I figured it's quite difficult to use LightGBM/XGboost and NN is taking over. Then I took a break from this competition to participate in a CV competition.  I rejoined the competition 13 days before the competition ends. There were already many great kernels and diverse methods shared in the notebooks section, I went through some of them and decided to tune some parameters and change the model architecture. I then teamed up with @nurmannaz which a public leaderboard of 1811. The 1811 submission itself is very important for further development of my models using pseudo labels; however, since this is @nurmannaz 's first kaggle competition, he wasn't aware that there's a private dataset so the submission itself is invalid. I had to rebuild everything from scratch and it was a lot of work. I began to train models and write inferences.

## Models
I came up with a method modified from https://arxiv.org/abs/1507.06228. It can be a good replacement of simple dense layer, implemented below:
```python
class Dense_Alternative(nn.Module):
    def __init__(self, h1, h2, f):
        super().__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.Linear(h1, h2)
        self.linear = nn.Linear(h1, h2)
        self.switch = nn.Linear(h1, h2)
        self.f = f

    def forward(self, x):
        switch = torch.sigmoid(self.gate(x))
        nonlinear = self.f(self.nonlinear(x))
        linear = self.linear(x)
        x = (1 - switch) * linear + switch * nonlinear 
        return x
```
It allows the neural network to decide the weight between linear and nonlinear units,
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5168115%2F9df6f34914348fa6cf02f5e5d408f1c3%2Fdense_alt.png?generation=1606788885682182&alt=media)

## Validation Strategies
I used both Multilabel Stratified Kfold and @cdeotte 's fold based on drug ID. I wished two different validation schemes can result in more stable ensemble. I still don't know if it helped or not.

## Pseudo Labels(PL)
PL is basically using unseen data's prediction as training data. Note that we should not use any of the samples in the unseen data for validations. To my understanding, there are two ways of using pseudo labels, one is to use the entire predictions, which is what I did in this competition; the other is to use soft labels based on probability, e.g. selecting samples with a certain probability threshold. I think PL is a very important part of my solution as it improves my CV and LB significantly. Moreover, I noticed the quality of the pseudo labels really matter. A 1810 submission can improve my CV a lot more than a 1850 submission. There is some potential risks of using pseudo labels, but I will not dive into this here. Namely, making wrong predictions wronger. 

An example usage of using pseudo label. In this case I first pretrained on all targets and then when I am finetuning, I add the pseudo labels.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5168115%2F288261cf066410c82f19681608e14dde%2Fpl_Explained.png?generation=1606791330584992&alt=media)

## Result
I tried to get more diverse models regardless of public LB and I took the mean of my selected models. It resulted in our best submission. I wish I had a bit more time. It was a rush to get all these done in twoo weeks![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5168115%2F12367d07d3f18922e5dc6c3b0d27fd4d%2Fscore.png?generation=1606789822769643&alt=media)

## Fin
Thanks for reading this. Please comment below if you have any questions. This solution is a lot worse than the best solutions but I hope it helped. I look forward to reading all your solutions.
