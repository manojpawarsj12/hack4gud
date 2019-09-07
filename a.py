import datetime
name="hello"
x = datetime.datetime.now()
b = x.strftime("%m/%d/%Y, %H:%M:%S")
name=name+" "+b
print(b)
print(type(b))
print(name)