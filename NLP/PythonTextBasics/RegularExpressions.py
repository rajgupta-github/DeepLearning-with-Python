import re

text = "The agent's phone number is 408-555-1234. Call soon!"
print('phone' in text)

pattern = 'phone'
match = re.search(pattern,text)
print(match)
print(match.span())
print(match.span()[0])
print(match.span()[1])

pattern = "NOT IN TEXT"
match = re.search(pattern,text)
print(match)

# But what if the pattern occurs more than once?
# Notice it only matches the first instance. If we wanted a list of all matches, we can use .findall() method:

text = "my phone is a new phone"
match = re.search("phone",text)
print(match.span())


matches = re.findall("phone",text)
print(matches)
print(len(matches))

for match in re.finditer("phone",text):
    print(match.span())
print(match.group())

