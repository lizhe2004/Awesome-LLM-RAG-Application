# Prompting vs RAGs vs Fine-tuning:

An important decision that every AI Engineer must make when building an LLM-based application.

To understand what guides the decision, let's first understand the meaning of these terms.

## 1️⃣ Prompting Engineering:

The prompt is the text input that you provide, based on which the LLM generates a response.

It's basically a refined input to guide the model's output.

The output will be based on the existing knowledge the LLMs has.

## 2️⃣ RAGs (Retrieval-Augmented Generation):

When you combine prompt engineering with database querying for context-rich answers, we call it RAG.

The generated output will be based on the knowledge available in the database.

## 3️⃣ Finetuning

Finetuning means adjusting parameters of the LLM using task-specific data, to specialise in a certain domain.

For instance, a language model could be finetuned on medical texts to become more adept at answering healthcare-related questions.

It's like giving additional training to an already skilled worker to make them an expert in a particular area.

Back to the important question, how do we decide what approach should be taken!

(refer the image below as you read ahead)
![](http://imgs.huahuaxia.net/picgo/20231227105350.png)

❗️There are two important guiding parameters, first one is Requirement of external knowledge, second is requirements of model adaptation.

❗️While the meaning of former is clear, model adaption means changing the behaviour of model, it's vocabulary, writing style etc.

For example: a pretrained LLM might find it challenging to summarize the transcripts of company meetings, because they might be using some internal vocabulary in between.

🔹So finetuning is more about changing structure (behaviour) than knowledge, while it's other way round for RAGs.

🔸You use RAGs when you want to generate outputs grounded to a custom knowledge base while the vocabulary & writing style of the LLM remains same.

🔹If you don't need either of them, prompt engineering is the way to go.

🔸And if your application need both custom knowledge & change in the behaviour of model a hybrid (RAGs + Finetuning) is preferred.


  提示工程、RAGs 与微调的对比：

这是每位搭建基于大语言模型（LLM）应用的 AI 工程师都面临的关键选择。

要理解这个决策的指导原则，我们首先得明白这些术语的含义。

1️⃣ 提示工程：

所谓提示，指的是你输入的文本，大语言模型就根据这个输入来生成回应。

这实际上是一种精确的输入方法，旨在引导模型产生相应的输出。

模型的输出将基于其已有的知识。

2️⃣ RAGs（检索增强生成）：

当你将提示工程与数据库查询结合，以获得含丰富上下文的答案时，这就是所谓的 RAG。

生成的输出将基于数据库中现有的知识。

3️⃣ 微调：

微调是指使用特定任务的数据调整大语言模型的参数，使其在某一领域内专业化。

比如，一个语言模型可以在医学文献上进行微调，从而更擅长回答健康护理相关的问题。

这就好比对一位已经技艺娴熟的工人进行额外培训，让他们在特定领域成为专家。

那么，我们如何决定采取哪种方法呢？

（阅读下文时请参考下面的图片）

❗️有两个关键的指导参数，一个是对外部知识的需求，另一个是模型适应性的需求。

❗️尽管前者的含义较为明确，模型适应性则意味着改变模型的行为、词汇、写作风格等。

例如，一个预训练的大语言模型可能在总结公司会议记录时遇到挑战，因为会议中可能穿插了一些特定的内部术语。

🔹因此，微调更多的是关于改变结构（行为）而非知识，而对于 RAGs 则正好相反。

🔸当你需要生成基于定制知识库的输出，同时保持大语言模型的词汇和写作风格不变时，你可以选择使用 RAGs。

🔹如果你不需要上述任一功能，那么提示工程就是你的选择。

🔸如果你的应用既需要定制知识又需要改变模型的行为，那么采用混合方案（RAGs + 微调）将是更佳选择。
