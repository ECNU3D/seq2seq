Main:	
if not FLAGS.data_path:
raise ValueError("Must set --data_path to PTB data directory")
#这是数据路径的判断
raw_data = ptb_reader.ptb_raw_data(FLAGS.data_path)
#利用ptb_raw_data这个函数来读取数据
#这个函数位于reader.py
#这个函数返回了 train_data,valid_data,test_data,word_dic,reverse_dic
global id_to_word,word_to_id
#定义 global变量 id_to_word,word_to_id
#也就是可以利用id查找word,或者是利用word查找id
#这边的word_to_id，将10000个单词做成词典，可以查阅每个单词在原始文章中
#出现的频数
#train_data,valid_data,test_data是原始文章中对应的每个单词的频数
#eg:一篇文章长929589，那么这3个list也长929589,并且每项对应的是频数
#这样的raw_data用于后续的信息熵的计算
train_data,valid_data,test_data,word_dict,reverse_dict = raw_data
id_to_word = reversoe_dict
word_to_id = word_dict
#将上述的变成名字更能懂的变量
#id_to_word和word_to_id
#print(word_dict.keys()[1],reverse_dict.keys()[1])
config = get_config()
#此处是得到配置
#配置有默认的small,medium,large
#在运行的时候可以通过 --model:""进行输入
#利用FLAGS的功能
eval_config = get_config()
eval_config.batch_size = 1
eval_config.num_steps = 1
#这边是eval_config
#并且将batch_size和num_steps赋值为1
#batch_size代表进行凸优化的时候每次所取的批次数目
#num_steps代表了默认的一句句子的长度
#在进行选取数据的时候，选取的是 batch_size * [num_steps个(词语) = 一句句子]
#这样的选取，同时，对应的label y 就是原文中每个单词对应的下一个单词构成的句子
#x 和 y 的 大小是相同的
#pdb.set_trace()
with tf.Graph() as_default(),tf.Session() as session:
#使用了一张默认度 default_graph
initializer = tf.random_uniform_initializewr(-config.init_scale,config.init_scale)
#tf.random_uniform_initializer(-x,x)
#定义训练用的神经网络模型
if FLAGS.interacvtive:
#定义交互模式，是网页还是终端
with tf.variable_scope("model",reuse = None,initializer=initializer):
PTBModel(is_training=False,config = eval_config,is_query = True)
 
tf.train.Saver().restore(session,FLAGS.data_path+FLAG.model+"-ptb.ckpt")
 
#with tf.variable_scope("model',reuse=True,initializer = initializer"):
#mvalid = PTBModel(is_training=False,config=confdig)
 
#valid_perplexity = run_epoch(session,mvalid,valid_data,tf.no_op())
#print("Valid Perplexity of trained model:%.3f"%(valid_perplexity))
 
#ptb.set_trace()
if FLAGS.interactive == "server":
#ptb_server.start_server(lamda x: run_input(session,x,30))
ptb_server.start_server(lamda i, x: run_input(session,x,30))
else:
entered_words = raw_input("enter your input")
while entered_words != "end":
print(run_input(session,entered_words,30));
entered_words = raw_input("Enter your input:")
sys.exit(0)
 
#上面的是进行预测的一部分，下面的是训练模型的一部分
 
with tf.variable_scope("mode",reuse = None,initializer=initializer):
m = PTBModel(is_training = True,config = config)
with tf.variable_scope("model",reuse = True,initializer = initializer):
mvalid = PTBModel(is_training = False,config = config)
mtest = PTBModel(is_training = False,config = eval_config)
 
tf.initialize_all_variable().run()
 
for i in range(config.max_epoch):
ir_decay = config.lr_decay**max(i - config.max_epoch,0.0)
m.assign_lr(session,config.learning_rate * lr_decay)
 
print("Epoch: %d Learning rate: %.3f"%(i+1,session.run(m.lr)))
train_perplexity = run_epoch(session,m,train_data,m.train_op,verbose= True)
 
print("Epoch: %d Train Perplexity: %.3f"%(i+1,train_perplexity))
 
valid_perplexity = run_epoch(session,mvalid,valid_data,tf.no_op())
print("Epoch: %d Valid Perplexity: %.3f"%(i+1,valid_perplexity))
 
print("Saving model")
tf.train_Saver().save(session,FLAGS.model+"-ptb.ckpt")
 
 
test_perplexity = run_epoch(session,mtest,test_data,tf.np_op())
print("Test Perplexity: %.3f"%test_perplexity)
print("Training Complete, saving model ... ")
model_path = os.path.join(FLAGS.data_path,"-ptb.ckpt")
tf.train.Saver().save(session,model_path)





在这边就需要讲一下model是怎么样的
需要说明的是python2和python3某些语句是不一样的
tensorflow的不同版本，语句也是不同的，这里用的是旧代码，但是新的代码我已经改完了
class PTBModel(object):
"""The PTB model."""
def __init___(self,is_training,config,is_query=False,is_generative=False):
self.batch_size = batch_size = config.batch_size
#获取configure中的batch_size
self.num_steps = num_steps = config.num_steps
#获取configure中的num_steps
size = config.hidden_szie
#获取configure中的hidden_size
#同时这边我需要说明，通过查阅了很多很多资料
#最终我知道这个hidden_size既是一层lstm 的 cell 数量
#同时也是一个单词(word)所对应的 word2vec(embedding)
#的维度大小
#也就是说一次向lstm中输入一个单词对应的embedding
vicab_size = config,vocab_size
#vocab_size就是词典的大小，为10000
#one input word is projected into hidden_size space using embeddings，
#using input word is projected into hidden_size space using embedding
#很不幸，官方文档是没有这些注释的
self._input_data = tf.placeholder(tf.int32,[batch_size,num_steps])
#需要输入的x
self._prior_output =tf.placeholder(tf.int32,[batch_size,size])
self._targets = tf.placeholder(tf.int32,[batch_size,num_steps])
#需要比对的y (在nlp的领域称其为 target)
 
#Slightly better results can be obtained with forget gate biases
#initialized to 1 but the hyperparameters of the model would need to be
#different than reported in the paper
#上面就是一些优化的方法
lstm_cell = rnn_cell.BasicLSTMCell(size,forget_bais =0.0)
# lstm一层的初始化，一层的cell个数为 size = hidden_size
# forget_bias = 0.0 遗忘系数为 0 
# 这个地方需要优化
if is_training and config.keep_prob < 1:
lstm_cell = rnn_cell.DropoutWrapper(
lstm_cell,output_keep_prob=config.keep_prob)
#这边的keep_prob是一个关于dropout方法的参数赋值
#也是可以优化，对宏观浏览lstm网络不造成影响
cell = rnn_cell.MultipleRNNCell([lstm_cell]*config.num_layers)
#mutipleRNNCell 扩大层数，变成2层
self._intial_state = cell.zero_state(batch_size,tf.float32)
#因为lstm每个时刻都有一个state的存在，t时刻的state是由t-1时刻产生的
#lstm 有 c,h,x三个输入
#x为输入，c为t-1时刻输出的状态，h为t-1时刻的输出
#但是这里为什么batch_size，暂时还没懂
#if not is_generative:
with tf.device("/cpu:0"):
                     embedding = tf.get_variable(
                                          "embedding", [vocab_size, size], dtype=data_type())
#创建embedding variable
#每个单词的embedding 为 size 维度
                      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
#将输入的batch_szie * num_steps大小的word 变成 batch_size * num_steps 个embedding
#每个embedding大小是size = hidden_size
 
            #else:
            #        inouts = tf.reshape(self._prior_output,[1,1,size])
 
            if is_training and not is_query and is_generative and config.keep_prob<1:
                    inputs = tf.nn.dropout(inputs,config.keep_prob)
# 如果此时正在进行train的话，再对Input进行一个dropout的修饰
#防止过拟合
#就是随机将cell取消
            #Simplified version of tensorflow.models.rnn.rnn.py's rnn().
            #This builds an unrolled LSTM for tutorial purposes only.
            #In general,use the rnn() or state_saving_rnn() from rnn.py.
            #
            #The alternative version of the code below is:
            #
            #from tensorflow.models.rnn import rnn
            #input = [tf.squeeze(input_,[1])
            #                        for input_ in tf.split(1,num_steps,inputs)]
            #outputs, states = rnn.rnn(cell,inoputs,initial_state=self._initial_state)
            outputs = []
            state = []
            state = self._initial_state
#输出和初始状态的定义
            with tf.variable_scope("RNN"):
                    for time_step in range(num_steps):#开始读入一句句子，一批batch的句子，一个单词一个单词读入
                            if time_step > 0: tf.get_variable_scope().reuse_variables()
#在第一个声明lstm结构中使用的变量，在之后的时刻都需要复用之前定义好的变量
                            (cell_output,state) = cell(inputs[:,time_steps,:],state)
#新的输出和状态
                            outputs.append(cell_output)
#将新的输出进行append
                            states.append(state)        
#将新的state进行append
            #output dimension is batch_sizeXhidden_size
            outputs = tf.concat(1,outputs)
            outut = tf.reshape(outputs,[-1,size])
#将输出队列展开成 [batch,hidden_szie*num_steps]
#再reshape成batch_size*num_steps，hidden_Szie 的形状
            #output tf.reshape(tf.concat(1,outputs),[-1,size])
            #logit dimenskion is batch_sizeX
            logits =tf.nn.xw_plus_b(
                                                    output,
                                                    tf.get_variable("softmax_w",[size,vocable_size]),
                                                    tf.get_variable("softmax_b",[vocable_size]))
#将lstm中得到的输出再经过一个全连接层得到最后的预测效果，最终的预测结果
#在每一个时刻上都是一个长度为vocab_size的数组，经过softmax层之后表示下一个位置
#不同单词的概率
#大小为vocable_size,在经过soft_max函数之后，为每个词在这个位置出现的概率
#就是所谓的perplexity
            self._logits = logits
            self._outputs = outputs
            self._output = output
            self._inputs = inputs
            self._final_state = states[-1]
 
            if is_query or is_generative:#如果是在进行查询预测，则不需要计算perplexity
                    #slef._loigts = tf.matmul(output,tf.get_variable("softmax_w"))+tf.get_variable("softmax_b")
                    probs = tf.nn.softmax(logits)
                    self._probs = probs
                    top_k = tf.nn.top_k(probs,29)[1]
                    self._top_k = top_k
                    return
            else:#反之进行loss function 交叉熵的计算进行优化
#tensorflow这边使用的是一个skip-gram的模型
#skip-gram是通过当前词去预测上下文，也就是target -> context
#具体说明在tensorflow官方文档之中
#真的写的挺差的…
# 
#下面的sequence_loss_by_example是特别的方法，我这边不赘述，在新版的tensorflow中语句也不同
#之前的BasicLSTMCell的语句也不同，我在本地也进行了修改
                    loss = seq2seq.sequence_loss_by_example([logits],
                                                                                            [tf.reshapeO(self._targets,[-1])],
                                                                                            [tf.ones([batch_size * num_steps])],
                                                                                            vocab_size)
 
            if not is_training:
                    return
#如果不是在进行训练，那么就返回
#下面是一些训练模型的方法 optimizer和学习效率的定义
            self._lr = tf.Variable(0,0,trainable = False)
            tvars = tf.trainable_variables()
            grads,_ = tf.clip_by_global_norm(tf.gradients(cost,tvars),
                                                                    config.max_grad_norm)
            optimizer = tf.train.GrdientDescentoptimizer(self.lr)
            self._train_op = optimizer.apple_gradients(zip(grads,tvars))
 
 
    def assign_lr(self,session,lr_value):
            session.run(tf.assign(self.lr,lr_value))
    @property
    def input_data(self):
            return self._input_data
    
    @property
    def targets(self):
            return self._targets
 
    @property
    def initial_state(self):
            return self._initial_state
    
    @property
    def output(self):
            return self._output
    
    @property
    def outputs(self):
            return self._outputs
    
    @property
    def prior_output(self):
            return self._prior_output
    
    @property
    def inputs(self):
            return self._inputs
    
    @property
    def logits(self):
            return self._logits
    
 
    @property
    def cost(self):
            return self._cost
    
    @property
    def top_k(self):
            return self._top_k
    
 
    @property
    def probs(self):
            return self._probs
    
    @property
    def final_state(self):
            return self._final_state
    
    @property
    def lr(self):
            return self._lr
    
    @property
    def train_op(self):
            return self._train_op



这边是run epoch
#这边比较直接，不太需要注释了
def run_epoch(session,m,data,eval_op,verbose = False):
"""Runs the model on the given data"""
epoch_size = ((len(data)//m.batch_size)-1)//m.num_steps
##the epoch_size here equals to the number of iteration!
start_time = time.time()
costs = 0.0
iters = 0
state = m.initial_state.eval()
for step,(x,y) in enumerate(ptb_reader.ptb_iterator(data,m.batch_size,m.num_steps)):
cost,state,inputs,output,outputs,__ = session.run(
 
[m.cost,
 m.final_state,
 m.inputs,
 m.output,
 m.outputs,
 eval_op],
 {m.input_data:x,
 m.targets:y,
 m.initiual_state:state})
#print(tf.shape(y).dims)
#print(tf.shape(output))
costs += cost
iters +=m.num_steps
if verbose and steps % (epoch_sizw//10)==10:
print("%.3f perplexity: %.3f speed: %.0f wps"%
(step *1.0/epoch_size,np.exp(costs/iters),
iters*m.batch_size/(time.time()-start_time)))
 
tavrs = tf.trainable_variables()
print("printing all traiinable vairable for time steps",m.num_steps)
for tavr in tavrs:
print(tvar.name,tavr.initialized_value())
return np.exp(costs/iters)

