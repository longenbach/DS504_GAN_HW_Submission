# DS504_GAN_HW_Submission

## Generator:
For the generator I kept the design farily simple.   

```python
g = Sequential()
g.add(Dense(256,input_dim = z_dim,activation='relu'))
g.add(Dense(512,activation='relu'))
g.add(Dense(1024,activation='relu'))
g.add(Dense(784, activation="tanh")) 
g.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
```

## Discrinimator:
For the generator I kept the design farily simple.   
```python
d = Sequential()
d.add(Dense(1024, input_dim=784,activation='relu'))
d.add(Dropout(rate=0.25))
d.add(Dense(512,activation='relu'))
d.add(Dropout(rate=0.25))
d.add(Dense(256,activation='relu'))
d.add(Dropout(rate=0.25))
d.add(Dense(1, activation="sigmoid")) 
d.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
```
## Training:


![ACE](img/ACE_server.png)
![Loss](img/GAN__loss.png)







