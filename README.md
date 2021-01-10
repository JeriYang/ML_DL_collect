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

# 4.RandomForest、GBDT、XGBoost
## 本质原理
    1) RandomForest(随机森林)
    随机森林是一个用随机方式建立的，包含多个决策树的集成分类器。Bagging:从原始的数据集中采取有放回的抽样
    
    2) GBDT(Gradient Boost Decision Tree，梯度提升树)
    GBDT是以决策树为基学习器的迭代算法，注意GBDT里的决策树都是回归树而不是分类树。
    Boosting根据错误率来取样（Boosting初始化时对每一个训练样例赋相等的权重1／n，然后用该算法对训练集训练t轮，
    每次训练后，对训练失败的样例赋以较大的权重），因此Boosting的分类精度要优于Bagging。
    
    3) XGBoost(eXtreme Gradient Boosting)
    与GBDT本质类似，GBDT是机器学习算法，XGBoost是该算法的工程实现。
    GBDT采用的是数值优化的思维, 用的最速下降法去求解Loss Function的最优解, 其中用CART决策树去拟合负梯度, 用牛顿法求步长.
    XGboost用的解析的思维, 对Loss Function展开到二阶近似, 求得解析解, 用解析解作为Gain来建立决策树, 使得Loss Function最优.

    
    4) XGB并行原理
    注意xgboost的并行不是tree粒度的并行，xgboost也是一次迭代完成才能进行下一次迭代的（第t次迭代的代价函数里面包含了前面t-1次迭代的预测值）。
    xgboost的并行是在特征粒度上的。我们知道，决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点），
    xgboost在训练之前，预先对数据进行排序，然后保存block结构，后面的迭代中重复的使用这个结构，大大减小计算量。
    这个block结构也使得并行称为了可能，在进行节点的分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，那么各个特征的增益计算就可以开多线程进行。
    为了降低排序成本，xgboost将数据存储在内存单元中，我们称之为块（block）。每个块中的数据以压缩列（CSC）格式存储，每列按相应的特征值排序。
    此输入数据布局仅需要在训练之前计算一次，并且可以在以后的迭代中重复使用，大大的减少了计算量。这个block结构使得并行变成了可能。XGBoost 也支持Hadoop实现

## L1,L2正则化的理解
    L1正则化和L2正则化虽然都可以控制过拟合，但它们的效果并不相同。当正则化强度逐渐增大（即C逐渐变小）， 参数 的取值会逐渐变小，但L1正则化会将参数压缩为0，L2正则化只会让参数尽量小，不会取到0。
    在L1正则化在逐渐加强的过程中，携带信息量小的、对模型贡献不大的特征的参数，会比携带大量信息的、对模型 有巨大贡献的特征的参数更快地变成0，所以L1正则化本质是一个特征选择的过程，掌管了参数的“稀疏性”。L1正 则化越强，参数向量中就越多的参数为0，参数就越稀疏，选出来的特征就越少，以此来防止过拟合。因此，如果 特征量很大，数据维度很高，我们会倾向于使用L1正则化。由于L1正则化的这个性质，逻辑回归的特征选择可以由 Embedded嵌入法来完成

## 学习链接
1). [随机森林和GBDT](https://zhuanlan.zhihu.com/p/37676630)<br>
2). [XGBoost和GBDT的优缺点及XGBoost可并行的原因](https://blog.csdn.net/GFDGFHSDS/article/details/104595261)<br>
3). [XGB详解](https://www.cnblogs.com/mantch/p/11164221.html)
详细原理和推导：<br>
1).[（一）提升树模型：GBDT原理与实践](https://blog.csdn.net/anshuai_aw1/article/details/82888222)
2).[（二）提升树模型：Xgboost原理与实践](https://blog.csdn.net/anshuai_aw1/article/details/82970489#_604)
3).[Xgboost系统设计：分块并行、缓存优化和Blocks for Out-of-core Computation](https://blog.csdn.net/anshuai_aw1/article/details/85093106)
4).[（三）提升树模型：Lightgbm原理深入探究](https://blog.csdn.net/anshuai_aw1/article/details/83659932)
# 5.RNN,LSTM,GRU
## 本质原理
    1).RNN：将当前RNN单元的输入和前一个RNN单元输出的Hidden State组合起来，经过一个Tanh激活函数，生成当前单元的Hidden State。
    2).LSTM: Forget Gate决定哪些历史信息要保留；
             Input Gate决定哪些新的信息要添加进来；
             Output Gate决定下一个Hidden State要携带哪些历史信息。
    3).GRU:只有两个Gates: Reset Gate和Update Gate
           一个用于决定哪些信息用于输入，一个用于决定下一个Hidden State要携带哪些历史信息。

## 学习链接
1).[动图详解LSTM和GRU](http://www.banbeichadexiaojiubei.com/index.php/2020/06/26/%E5%8A%A8%E5%9B%BE%E8%AF%A6%E8%A7%A3lstm%E5%92%8Cgru/)
