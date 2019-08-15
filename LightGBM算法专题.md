# LightGBM算法梳理

## 一、LightGBM的起源
- LightGBM 是一个梯度 boosting 框架, 使用基于学习算法的决策树. 它是分布式的, 高效的, 装逼的, 它具有以下优势:

	+ 速度和内存使用的优化
	+ 减少分割增益的计算量
	+ 通过直方图的相减来进行进一步的加速
	+ 减少内存的使用 减少并行学习的通信代价
- 稀疏优化
- 准确率的优化
	+ Leaf-wise (Best-first) 的决策树生长策略
	+ 类别特征值的最优分割
- 网络通信的优化
- 并行学习的优化
	+ 特征并行
	+ 数据并行
	+ 投票并行
- GPU 支持可处理大规模数据


## 二、Histogram VS pre-sorted

- Histogram算法并不是完美的。由于特征被离散化后，找到的并不是很精确的分割点，所以会对结果产生影响。但在实际的数据集上表明，离散化的分裂点对最终的精度影响并不大，甚至会好一些。原因在于decision tree本身就是一个弱学习器，采用Histogram算法会起到正则化的效果，有效地防止模型的过拟合。
- 时间上的开销由原来的O(#data * #features)降到O(k * #features)。由于离散化，#bin远小于#data，因此时间上有很大的提升。
- Histogram算法还可以进一步加速。一个叶子节点的Histogram可以直接由父节点的Histogram和兄弟节点的Histogram做差得到。一般情况下，构造Histogram需要遍历该叶子上的所有数据，通过该方法，只需要遍历Histogram的k个捅。速度提升了一倍。

## 三、leaf-wise VS level-wise

- 直方图算法的基本思想：先把连续的浮点特征值离散化成k个整数，同时构造一个宽度为k的直方图。遍历数据时，根据离散化后的值作为索引在直方图中累积统计量，当遍历一次数据后，直方图累积了需要的统计量，然后根据直方图的离散值，遍历寻找最优的分割点。

- 带深度限制的Leaf-wise的叶子生长策略

- Level-wise过一次数据可以同时分裂同一层的叶子，容易进行多线程优化，也好控制模型复杂度，不容易过拟合。但实际上Level-wise是一种低效算法，因为它不加区分的对待同一层的叶子，带来了很多没必要的开销，因为实际上很多叶子的分裂增益较低，没必要进行搜索和分裂。

- Leaf-wise则是一种更为高效的策略：每次从当前所有叶子中，找到分裂增益最大的一个叶子，然后分裂，如此循环。因此同Level-wise相比，在分裂次数相同的情况下，Leaf-wise可以降低更多的误差，得到更好的精度。

- Leaf-wise的缺点：可能会长出比较深的决策树，产生过拟合。因此LightGBM在Leaf-wise之上增加了一个最大深度限制，在保证高效率的同时防止过拟合。


- 当生长相同的叶子时，Leaf-wise 比 level-wise 减少更多的损失。

## 四、特征并行和数据并行

- LightGBM 还具有支持高效并行的优点。LightGBM 原生支持并行学习，目前支持特征并行和数据并行的两种。

- 特征并行的主要思想是在不同机器在不同的特征集合上分别寻找最优的分割点，然后在机器间同步最优的分割点。
- 数据并行则是让不同的机器先在本地构造直方图，然后进行全局的合并，最后在合并的直方图上面寻找最优分割点。

- LightGBM 针对这两种并行方法都做了优化：
	- 在特征并行算法中，通过在本地保存全部数据避免对数据切分结果的通信；
	- 在数据并行中使用分散规约 (Reduce scatter) 把直方图合并的任务分摊到不同的机器，降低通信和计算，并利用直方图做差，进一步减少了一半的通信量。基于投票的数据并行则进一步优化数据并行中的通信代价，使通信代价变成常数级别。在数据量很大的时候，使用投票并行可以得到非常好的加速效果。

## 五、顺序访问梯度

- 预排序算法中有两个频繁的操作会导致cache-miss，也就是缓存消失（对速度的影响很大，特别是数据量很大的时候，顺序访问比随机访问的速度快4倍以上 ）。

- 对梯度的访问：在计算增益的时候需要利用梯度，对于不同的特征，访问梯度的顺序是不一样的，并且是随机的对于索引表的访问：预排序算法使用了行号和叶子节点号的索引表，防止数据切分的时候对所有的特征进行切分。同访问梯度一样，所有的特征都要通过访问这个索引表来索引。
这两个操作都是随机的访问，会给系统性能带来非常大的下降。

- LightGBM使用的直方图算法能很好的解决这类问题。首先。对梯度的访问，因为不用对特征进行排序，同时，所有的特征都用同样的方式来访问，所以只需要对梯度访问的顺序进行重新排序，所有的特征都能连续的访问梯度。并且直方图算法不需要把数据id到叶子节点号上（不需要这个索引表，没有这个缓存消失问题）

## 六、支持类别特征

- 传统的机器学习一般不能支持直接输入类别特征，需要先转化成多维的0-1特征，这样无论在空间上还是时间上效率都不高。LightGBM通过更改决策树算法的决策规则，直接原生支持类别特征，不需要转化，提高了近8倍的速度。

## 七、应用场景

- 强大的兼容性，分类和回归问题都可以使用

## 八、sklearn参数
class lightgbm.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1, silent=True, importance_type='split', **kwargs)


- boosting_type：优化方法
- num_leaves：决策树最大叶结点数量
- max_depth：决策树最大深度
- learning_rate：学习率
- n_estimators：基础学习器数量
- subsample_for_bin：确定binning边界时时考虑的样本数量
- objective：优化函数
- class_weight：类别权重
- min_split_gain：节点继续分裂最小损失函数下降
- min_child_weight：叶结点最小样本权重
- min_child_samples：叶结点最小样本数量
- subsample：样本抽样
- subsample_freq：样本抽样
- colsample_bytree：特征抽样
- reg_alpha：正则化变量
- reg_lambda：正则化变量
- random_state：随机种子
- n_jobs：控制并行
- silent：控制输出
- importance_type：特征重要程度计算方式

## 九、CatBoost(了解)


- 它自动采用特殊的方式处理类别型特征（categorical features）。首先对categorical features做一些统计，计算某个类别特征（category）出现的频率，之后加上超参数，生成新的数值型特征（numerical features）。这也是我在这里介绍这个算法最大的motivtion，有了catboost，再也不用手动处理类别型特征了。
- catboost还使用了组合类别特征，可以利用到特征之间的联系，这极大的丰富了特征维度。
- catboost的基模型采用的是对称树，同时计算leaf-value方式和传统的boosting算法也不一样，传统的boosting算法计算的是平均数，而catboost在这方面做了优化采用了其他的算法，这些改进都能防止模型过拟合。

## 参考链接
1、https://www.jianshu.com/p/49ab87122562



