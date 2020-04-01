# 杰克租车问题(Jack's Car Rental)

 杰克管理着一个全国汽车租赁公司的两个地点。每天，一些顾客到每个地点租车。 如果杰克有一辆车可以用来出租，那么他将车租出去并且得到租车公司的$10$美元报酬。 如果他在这个地点没有车，那么这次生意机会就失去了。汽车被送回来之后就可以被租出去。 为了确保人们需要车子的时候有车可以租，杰克可以在晚上将车子在两个地点之间转移，每转移一辆车需要花费$2$美元。 我们假设需要车子的数量与返回车子的数量是泊松随机变量，也就是说数量 $n$ 的概率 是 $\frac{\lambda^n}{n!}e^{-\lambda}$，$\lambda$ 是期望值。 假设第一个和第二个地点对于租借需求 $\lambda$ 是$3$和$4$，返回数量是$3$和$2$。 为了简化问题，我们假设每个位置不会超过$20$辆车（任何多余的车都将会被返回租赁公司，从问题中消失）， 并且最多五辆车可以在同一晚上从一个地点转移到另一个地点。 我们将衰减因子设置为$\lambda=0.9$，将这个问题当作连续有限马尔可夫决策过程（MDP）[^chapter4]。

# 解答
## 建模
将这个问题当作连续有限MDP，时间步骤是天数。首先要明确“状态”和“动作”是什么。 **状态**是每天结束是在每个位置剩余车子的数量，**动作**是每晚将车子在两个地点转移的净数量。然后用**动态规划**的方法解。 

计算机程序中，用动态规划来解的问题通常有两大特征：
 - 问题可分解成子问题，子问题独立；
 - 子问题重叠：这样才可以用一套代码来解，否则这种分解就没有意义了。

在强化学习中，动态规划的核心思想是用价值函数来结构化地组织对最优策略的搜索。一旦得到了**最优价值函数**$v_*$或**最优动作价值函数**$q_*$，得到最优策略就很容易了。动作价值函数有效地缓存了所有一步一步搜索的结果。 它提供最优的期望长期回报作为本地并立即可用于每个状态—动作对的值。 因此，代表状态-动作对的功能而不仅仅是状态的代价， 最优动作-价值函数允许选择最优动作而不必知道关于可能的后继状态及其值的任何信息，即不必要了解环境的动态。这样就把最优策略寻找问题分解成一个个独立、重叠的“状态-动作对”的子问题。

整个MDP中，状态之间的转移，是固定的（不依赖于动作，而是服从泊松到达），所以首先可以构建状态之间的转移概率表。每个租车点晚上的可能状态（汽车数量）为$s\in[0,20]$，共$21$种。两个租车点则对应着交叉遍历$21*21=441$种情况（两个状态向量的外积/kronecker积）。所以这个状态表“Tp”是一个$21*21=441$的矩阵。状态转移时，对应的收益也是固定的，记录在表“R”中。
从状态$s$，历经租车、还车，转移到新的状态。很容易算出转移到下一个状态的概率。

 ```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns; sns.set()

rent_income = 10
move_cost = 2
discount = 0.9
MAX_CAR_GARAGE = 20
MAX_CAR_MOVE = 5
MAX_CAR_USE = MAX_CAR_GARAGE+MAX_CAR_MOVE
lambda_rent = [3,4]
lambda_return = [3,2]
accurate = 1e-6

Tp = np.zeros(2*21*21).reshape(2,21,21) #状态转移矩阵，21*21个状态
R = np.zeros(2*21).reshape(2,21) #一步收益，2个地点，21个状态

def possion_prob(lam,n):
    return np.exp(-lam) * (lam**n) / np.math.factorial(n)
    
def trans_prob(s, garage):
    """
    计算每个租车点的状态转移概率
    """
    for r in range(0,MAX_CAR_GARAGE+1): # 当天租出去的车数量，可以取到正无穷，但指数衰减取到MAX_CAR_GARAGE足够保证精度
        p_rent = possion_prob(lambda_rent[garage], r) #租出去r辆车的概率
        if p_rent<accurate: #精度限制
            return
        rent = min(s, r) #租车数不可能大于库存数
        R[garage, s] += p_rent * rent_income * rent #租车收益
        for ret in range(0,MAX_CAR_GARAGE+1): #当天还车数量ret
           p_ret = possion_prob(lambda_return[garage], ret) #还ret辆车的概率
           if p_ret<accurate: #精度限制
               continue
           s_next = min(s-rent+ret, MAX_CAR_GARAGE) #下一步状态：租车+还车后的租车点汽车数量
           Tp[garage, s, s_next] += p_rent * p_ret #状态转移概率

def init_trans_prob():
    """
    计算状态转移概率
    """
    for i in range(0,MAX_CAR_GARAGE+1):
        trans_prob(i, 0)
        trans_prob(i, 1)  
```
## 策略评估
对任意策略$\pi$，都可以写出其状态价值函数$v_{\pi}$，使用迭代的方法来解，称之为**迭代策略评估**。策略评估得到价值函数表“V”。迭代策略评估中，对每个状态采用相同的操作：*根据给定策略，得到所有可能的单步转移后的即时收益和每个后继状态的旧的价值函数，利用这二者的期望来更新状态的价值函数*。这种方法为**期望更新**，可以有不同形式。下面是使用“状态”进行期望更新的代码。注意只有一个价值函数表“V”，新的价值一个个被计算出来使用，“就地”更新。

 ```python
V = np.zeros(21*21).reshape(21,21)

def policy_evalue():
    delta = 0
    for i in range(0,MAX_CAR_GARAGE+1):
        for j in range(0,MAX_CAR_GARAGE+1): #对每个状态(i,j)
            v = V[i,j]
            temp_v = 0
            for m in range(0,MAX_CAR_GARAGE+1):
                for n in range(0,MAX_CAR_GARAGE+1): #可能的后继状态(m,n)
                    # 转移概率Tp*(沿途预期的奖励R+预期的后继状态的（衰减）值V
                    temp_v += Tp[0,i,m]*Tp[1,j,n]*(R[0,i] + R[1,j] + discount*V[m,n]) 
            V[i,j] = temp_v
            delta = max(delta, np.abs(v-V[i,j]))
    return delta
```
上面的写法里面没有考虑“动作”的影响，下面把动作考虑进去，用“状态-动作”二元组来进行**期望更新**。
 ```python
V = np.zeros(21*21).reshape(21,21)
Action = np.zeros(21*21).reshape(21,21)

def policy_evalue2():
    delta = 0
    for i in range(0,MAX_CAR_GARAGE+1):
        for j in range(0,MAX_CAR_GARAGE+1): # 对所有可能状态
            v = V[i,j]
            a = Action[i,j] #动作
            V[i,j] = value_calculate(i,j,a) # 价值评估
            delta = max(delta, np.abs(v-V[i,j]))
    return delta

def value_calculate(i,j,a):
    """
    a: 从租车点0移到租车点1的汽车数量; a \in [-5,5]
    """
    if a>i:
        a = i #从租车点0移走的车数不可能大于库存
    elif a<0 and -a>j:
        a = -j #从租车点1移走的车数不可能大于库存
    ii = int(i - a)
    jj = int(j + a)
    ii = min(ii, MAX_CAR_GARAGE)
    jj = min(jj, MAX_CAR_GARAGE) # 移车动作后，状态从 (i,j) 变成 (ii,jj)
    temp_v = -np.abs(a) * move_cost # 移车代价
    for m in range(0,MAX_CAR_GARAGE+1):
        for n in range(0,MAX_CAR_GARAGE+1): #对所有后继状态(m,n)
            temp_v += Tp[0,ii,m]*Tp[1,jj,n]*(R[0,ii] + R[1,jj] + discount*V[m,n])
    return temp_v
```
可以写成矩阵运算的形式，计算效率高。其中用到了矩阵的Hadamard乘积和Kronecker乘积。

 ```python
def value_calculate_matrix(i,j,a):
    if a>i:
        a = i
    elif a<0 and -a>j:
        a = -j
    ii = int(i - a)
    jj = int(j + a)
    ii = min(ii, MAX_CAR_GARAGE)
    jj = min(jj, MAX_CAR_GARAGE) # after move_action, update (i,j) to (ii,jj)
    Tp_matrix = np.outer(Tp[0,jj], Tp[1,ii]) # Kronecker product
    V_matrix = discount*np.mat(V)
    V_matrix = np.add(V_matrix, np.transpose(R[0,ii]))
    V_matrix = np.add(V_matrix, R[1,jj])
    V_matrix -= np.abs(a) * move_cost
    V_matrix = np.multiply(Tp_matrix, V_matrix) # Hadamard product
    return np.sum(V_matrix)
 ```

## 策略更新
用贪心算法对每个状态找到最优动作。贝尔曼最优方程阐述了一个事实：最优策略下各个状态的价值一定等于这个状态下最优动作的期望回报。这样对**最优价值函数**$v_*$来说，贪心策略最优。定义*$v_*$的意义就在于，我们可以将长期（全局）回报期望值转化为每个状态对应的一个当前局部变量的计算。一次单步搜索就可以产生长期（全局）最优动作序列。
$$
\begin{split}
v_*(s)&=\max_{a\in \mathcal{A}(s)} q_{\pi_*}(s,a) \\
&= \max_a \mathbb{E}_{\pi_*}[ G_t | S_t=s, A_t=a ] \\
&= \max_a \mathbb{E}_{\pi_*}[ R_{t+1}+\gamma G_{t+1} | S_t=s, A_t=a ] \\
&= \max_a \mathbb{E}_{\pi_*}[ R_{t+1}+\gamma v_*(S_{t+1}) | S_t=s, A_t=a ] \\
&= \max_a \sum_{s',r} p(s',r|s,a)[r+\gamma v_*(s')] \\
\end{split}
$$

 ```python
def action_greedy(i,j):
    """
    对状态(i,j)进行贪心算法选择最优动作
    """
    best_action = 0
    best_value = 0
    for a in range(-MAX_CAR_MOVE, MAX_CAR_MOVE+1):
        if a>i:
            continue
        elif a<0 and -a>j:
            continue
        val = value_calculate(i,j,a)
        if val>best_value+0.1:
            best_value = val
            best_action = a
    return best_action

def policy_improve():
    """
    对所有状态更新动作
    """
    stable_flag = True
    for i in range(0,MAX_CAR_GARAGE+1):
        for j in range(0,MAX_CAR_GARAGE+1):
            act_best = action_greedy(i,j)
            if act_best != Action[i,j]:
                Action[i,j] = act_best
                stable_flag = False
    return stable_flag
 ```

## 策略迭代
现在可以根据策略迭代算法计算最优策略了。
 ```python
init_trans_prob() #计算状态转移矩阵
stable = False
policies = []
while not stable: #策略迭代
    print "Evaluate Policies..."
    while 1: #策略评估
        delta = policy_evalue2() 
        if delta<0.1:
            print "Evaluate Finished!"
            break
    print "Improve Policies..."
    stable = policy_improve() #策略更新
    policies.append(Action.copy())
 ```
## 运行结果
最优策略后，收益曲面如下图所示。
![最优策略收益](https://upload-images.jianshu.io/upload_images/18876951-75f4955587fff4a9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

每次迭代后的策略下图所示。
![第1次迭代策略](https://upload-images.jianshu.io/upload_images/18876951-d8f1d279f26f6cf4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![第2次迭代策略](https://upload-images.jianshu.io/upload_images/18876951-f7e07f9f9f7b2d7c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![第3次迭代策略](https://upload-images.jianshu.io/upload_images/18876951-3ab0dd376574fdc2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![第4次迭代策略](https://upload-images.jianshu.io/upload_images/18876951-83e2102af04c0ed4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![第5次迭代策略](https://upload-images.jianshu.io/upload_images/18876951-ee1616d6c0b6da62.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

画图的代码附上如下：
 ```python
def plot_value1():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Income')
    aZ = []
    aX = []
    aY = []
    for i in range (MAX_CAR_GARAGE+1):
        for j in range (MAX_CAR_GARAGE+1):
            aX.append(i)
            aY.append(j)
            aZ.append(V[i, j])
    ax.set_ylabel('# of cars at location 1')
    ax.set_xlabel('# of cars at location 2')
    ax.scatter(aX, aY, aZ)  
  
def plot_value2():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(range(0,MAX_CAR_GARAGE+1), range(0,MAX_CAR_GARAGE+1))
    ax.scatter(X, Y, V) 
    
def print_policy(p,i=''):
    plt.figure()
    ticks = [0]+['']*(MAX_CAR_GARAGE-1)+[MAX_CAR_GARAGE]
    ax = sns.heatmap(p.astype(int),square=True,xticklabels=ticks,yticklabels=ticks)
    ax.set_title('Policy '+str(i))
    ax.set_ylabel('# of cars at location 1')
    ax.set_xlabel('# of cars at location 2')
    ax.invert_yaxis()
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.arange(MAX_CAR_MOVE*2+1)-MAX_CAR_MOVE)
    cbar.set_ticklabels(np.arange(MAX_CAR_MOVE*2+1)-MAX_CAR_MOVE)    
 ```

# 参考资料
[^chapter4]: [强化学习导论-第4章-动态规划](https://rl.qiwihui.com/zh_CN/latest/index.html)


