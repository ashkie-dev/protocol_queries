

while True:
    print('########################################')
    pt = input('Question: ')
    if pt.lower() == 'end':
        break
    response = chatbot(pt)
    print('Question:', pt)
    print('----------------------------------------')
    print('Answer: ')
    print(response)
