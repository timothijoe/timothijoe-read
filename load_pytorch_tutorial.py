# Tutorial
####################
I 三个核心函数
1. torch.save: 使用pickle
2. torch.load: 使用pickle
3. torch.nn.Module.load_state_dict() 

torch.nn.Module的模型中的可以学习的参数，bias, weight,被包含在模型的参数中，使用model.parameters()访问。
state_dict是一个简答你的python字典对象，其中映射每层网络的对应参数的张量。优化器 torch.optim也有包含优化器状态信息的state_dict，通常被当做超参数使用
####################
II 保存和加载模型的结论
Save: 
torch.save(model.state_dict(), PATH)
Load:
model = ModelClass(* args, ** kargs)
model.load_state_dict(torch.load(PATH))
model.eval()

注意， load_state_dict方法接受的是字典对象，不是保存对象的路径。
这意味着不可以直接model.load_state_dict(地址)
####################
III 保存或者加载常规的检测点，然后恢复训练

#将epoch, model, optimizer的state dict保存到字典中
torch.save({
    'epoch':epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}, PATH)

model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)
checkpoint = torch.laod(PATH)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['ep[och']
loss = checkpoint['loss']

model.eval()
# -or-
model.train()

请记住，在运行推断之前必须调用model.eval()将dropout和批处理标准化设置为评估模式。如果不这样做，就会产生不一样的推理结果
如果希望恢复训练，调用model.train()确保这些层处于训练模式

####################
IV 一个文件中保存多个模型
Save:
torch.save({
    'modelA_state_dict': modelA.state_dict(),
    'modelB_state_dict': modelB.state_dict(),
    'optimizerA_state_dict': optimizerA.state_dict(),
    'optimizerB_state_dict': optimizerB.state_dict(),
}, PATH)

Load:
modelA = TheModelAClass(*args, **kwargs)
modelB = TheModelBClass(*args, **kwargs)
optimizerA = TheOptimizerAClass(*args, **kwargs)
optimizerB = THeOptimizerBClass(*args, **kwargs)

checkpoint = torch.load(PARH)
modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizerA.laod_state_dict(checkpoint['optimizerA_state_dict'])

modelA.eval()
or 
modelA.train()



####################
V 使用不同的模型参数启动模型

modelB = TheModelClass(*args, **kwargs)
modelB.load_state_dict(torch.load(PATH), strict=False)
modelB.eval()
此时无论加载缺少某些键的局部state_dict,还是加载一个比正在使用的模型键还多的state_dict,都可以在load_state_dict()函数中将参数设定为False,以忽略不匹配的键。
也可以更正state_dict的参数名字，以匹配加载的模型的键


####################
VI 跨设备加载模型

Save on GPU, load on cpu

Save:
torch.save(model.state_dict(), PATH)

Load1:
device = torch.device('cpu')
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))
model = eval()

Load2:
device = torch.device('cuda')
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)
model = eval()

Load2:
device = torch.device('cuda')
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH), map_location='cuda:0')
model.to(device)
model = eval()
