myfile = open('test.txt')
myfile.seek(0)
print(myfile.read())

myfile.seek(0)
for line in myfile.readlines():
    print(line)

myfile = open("test.txt", "w+")
myfile.write('This is a new first line')
myfile.seek(0)
print(myfile.read())
myfile.close()

myfile = open('test.txt','a+')
myfile.write('\nThis line is being appended to test.txt')
myfile.write('\nAnd another line here.')
myfile.seek(0)
print(myfile.read())
myfile.close()

with open('test.txt','r') as txt:
    for line in txt:
        print(line, end='')  # the end='' argument removes extra linebreaks
