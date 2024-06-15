def factorial(n):
    if n < 0:
        return "Factorial is not defined for negative numbers"
    elif n == 0:
        return 1
    else:
        result = 1
        for i in range(1, n + 1):
            result *= i
        return "对应的阶乘为"+str(result)
while True:  #输入模块
    user_input = input("请输入非负整数（输入'z'退出）: ")  
    if user_input == 'z':  
        print("已退出输入。")  
        break  
    else:  
        print("你输入了:", user_input)
        try:  #捕捉错误模块
            int_input=int(user_input)
        except ValueError:
            print("Error!!!您输入的不是整数类型变量!!!")
        else:
            print(factorial(int_input))

            


        
