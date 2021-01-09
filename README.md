# 1. Attention机制
## 本质原理
    第一步： query 和 key 进行相似度计算，得到权值
    第二步：将权值进行归一化，得到直接可用的权重
    第三步：将权重和 value 进行加权求和
    PS:
    Attention 的思路简单，四个字“带权求和”就可以高度概括，大道至简。
    做个不太恰当的类比，人类学习一门新语言基本经历四个阶段：
    死记硬背（通过阅读背诵学习语法练习语感）
    ->提纲挈领（简单对话靠听懂句子中的关键词汇准确理解核心意思）
    ->融会贯通（复杂对话懂得上下文指代、语言背后的联系，具备了举一反三的学习能力）
    ->登峰造极（沉浸地大量练习）。
    这也如同attention的发展脉络，
    RNN 时代是死记硬背的时期，
    attention 的模型学会了提纲挈领，进
    化到 transformer，融汇贯通，具备优秀的表达学习能力，
    再到 GPT、BERT，通过多任务大规模学习积累实战经验，战斗力爆棚。

## 分类
![image](https://github.com/JeriYang/ML_DL_collect/blob/main/pic/attention_types.png)

## 学习链接
1). [Attention本质原理+3大优点+5大类型](https://medium.com/@pkqiang49/%E4%B8%80%E6%96%87%E7%9C%8B%E6%87%82-attention-%E6%9C%AC%E8%B4%A8%E5%8E%9F%E7%90%86-3%E5%A4%A7%E4%BC%98%E7%82%B9-5%E5%A4%A7%E7%B1%BB%E5%9E%8B-e4fbe4b6d030)

# 2. Transformers
## 本质原理
通过attention机制，和query, keys, values来计算向量的得分，然后选取得分高的向量。

## 学习链接
1). [多图带你读懂 Transformers 的工作原理](https://www.leiphone.com/news/201903/ELyRKiBJOx8agF1Q.html)

# 3. AR模型与AE模型
## 本质原理
    + AR语言模型(AutoRegressive LM)：
    只能获取单向信息，即只能前向读取信息并预测t位置的单词或者从后向读取信息并预测t位置的单词，
    却不能同时获取双向信息，代表例子是GPT，GPT2，XLNet，ELMO
    
    + AE语言模型(AutoEncoder LM):
    获取双向信息进行预测，如想要预测位置t的单词，
    既可以前向获取信息也可以后向获取信息，代表例子是Bert

## 各自优缺点
    AR LM的优点：比较擅长生成类任务
           缺点：只能获取单向信息，不能获取双向信息。
    
    AE LM的优点：可以获取双向信息，能同时看到预测位置的上文和下文
           缺点：如bert，它在预训练过程中会增加输入噪声，
           如对输入序列会随机mask掉一部分的单词，而在微调时却不会增加输入噪声，
           这种预训练-微调步骤中产生的差异，会产生一部分的人为误差

## 学习链接
1）. [AR模型与AE模型](https://www.cnblogs.com/mj-selina/p/12392839.html)
