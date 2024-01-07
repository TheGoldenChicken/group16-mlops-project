# group16: Premise-Hypothesis classification

We have chosen to work with **HuggingFace** as the main third-party software \\

Project work folder for group 19 working with huggingface transformers as primary third-party software. \\

Our project is centered around a premise-hypothesis classification problem posed by [https://huggingface.co/datasets/multi_nli](https://huggingface.co/datasets/multi_nli), 
where a model is given a hypothesis and a premise and needs to classify the hypothesis as either either somewhat unrelated, yet similar in nature (neural), 
as following logically  from the premise (entailment), or as actually contradicting the premise (contradiction). \\

For this task, we plan to use a pre-trained version of BERT, taken from HuggingFace. We will use transfer learning to fit it specifically to the actual problem. \\

Of course the dataset might change, we've found more than one example of the MNLI data we're using, several of them on the officla HuggingFace site, others on Kaggle. \\
This gives some opportunity to use dvc, even though the opoprtunity is totally ripe, so to speak.


We use this as inspiration for what the fuck we gots to do
https://medium.com/@anuranjana25/multiclass-classification-for-natural-language-inference-bbc6b9df1b10
