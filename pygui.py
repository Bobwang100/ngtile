# from tkinter import *
# root = Tk()
# root.title('我的第一个Python窗体')
# root.geometry('240x240') # 这里的乘号不是 * ，而是小写英文字母 x
# root.mainloop()

# from  tkinter import *
# root = Tk()
# lb = Label(root,text='我是第一个标签',\
#         bg='#d3fbfb',\
#         fg='red',\
#         font=('华文新魏',32),\
#         width=20,\
#         height=2,\
#         relief=RAISED)
# lb.pack()
# root.mainloop()

# from tkinter import  *
# root = Tk()
#
# lbred = Label(root,text="Red",fg="Red",relief=GROOVE)
# lbred.pack()
# lbgreen = Label(root,text="绿色",fg="green",relief=GROOVE)
# lbgreen.pack()
# lbblue = Label(root,text="蓝",fg="blue",relief=GROOVE)
# lbblue.pack()
# root.mainloop()

from tkinter import *

def run1():

     s = '%0.2f+%0.2f=%0.2f\n' % (a, b, a + b)
     txt.insert(END, s)   # 追加显示运算结果
     # inp1.delete(0, END)  # 清空输入
     # inp2.delete(0, END)  # 清空输入

def run2(x, y):
     a = float(x)
     b = float(y)
     s = '%0.2f+%0.2f=%0.2f\n' % (a, b, a + b)
     txt.insert(END, s)   # 追加显示运算结果
     # inp1.delete(0, END)  # 清空输入
     # inp2.delete(0, END)  # 清空输入

root = Tk()
root.geometry('920x480')
root.title('深度学习磁片检测系统')

lb1 = Label(root, text='A 数据准备', bg='red', font=('华文新魏', 14), relief=SUNKEN)
lb1.place(relx=0.05, rely=0.05, relwidth=0.15, relheight=0.1)

lb2 = Label(root, text='B 训练阶段', bg='#d3fbfb', font=('华文新魏', 14), relief=SUNKEN)
lb2.place(relx=0.3, rely=0.05, relwidth=0.15, relheight=0.1)

lb1 = Label(root, text='C 测试阶段', bg='#d3fbfb', font=('华文新魏', 14), relief=SUNKEN)
lb1.place(relx=0.55, rely=0.05, relwidth=0.15, relheight=0.1)

lb1 = Label(root, text='D 模型生成', bg='#d3fbfb', font=('华文新魏', 14), relief=SUNKEN)
lb1.place(relx=0.8, rely=0.05, relwidth=0.15, relheight=0.1)

logo = PhotoImage(file='logo1.png')
lb1 = Label(root, text='图片', image=logo, font=('华文新魏', 15))
lb1.place(relx=0.7, rely=0.7, relwidth=0.3, relheight=0.3)

# inp1 = Entry(root)
# inp1.place(relx=0.1, rely=0.2, relwidth=0.3, relheight=0.1)
# inp2 = Entry(root)
# inp2.place(relx=0.6, rely=0.2, relwidth=0.3, relheight=0.1)

# 方法-直接调用 run1()
btn1 = Button(root, text='数据选择', command=None)
btn1.place(relx=0.075, rely=0.2, relwidth=0.1, relheight=0.1)

btn1 = Button(root, text='数据增强', command=None)
btn1.place(relx=0.075, rely=0.35, relwidth=0.1, relheight=0.1)

btn1 = Button(root, text='TF数据生成', command=None)
btn1.place(relx=0.075, rely=0.5, relwidth=0.1, relheight=0.1)


# 训练阶段
btn1 = Button(root, text='网络/模型选择', command=None)
btn1.place(relx=0.325, rely=0.2, relwidth=0.1, relheight=0.1)

btn1 = Button(root, text='数据读取', command=None)
btn1.place(relx=0.325, rely=0.35, relwidth=0.1, relheight=0.1)

btn1 = Button(root, text='进行训练', command=None)
btn1.place(relx=0.325, rely=0.5, relwidth=0.1, relheight=0.1)


# 测试阶段
btn1 = Button(root, text='模型选择', command=None)
btn1.place(relx=0.575, rely=0.2, relwidth=0.1, relheight=0.1)

btn1 = Button(root, text='数据读取', command=None)
btn1.place(relx=0.575, rely=0.35, relwidth=0.1, relheight=0.1)

btn1 = Button(root, text='进行测试', command=None)
btn1.place(relx=0.575, rely=0.5, relwidth=0.1, relheight=0.1)


# 模型生成阶段
btn1 = Button(root, text='模型选择', command=None)
btn1.place(relx=0.825, rely=0.2, relwidth=0.1, relheight=0.1)

btn1 = Button(root, text='pb文件生成', command=None)
btn1.place(relx=0.825, rely=0.35, relwidth=0.1, relheight=0.1)


# 在窗体垂直自上而下位置60%处起，布局相对窗体高度40%高的文本框
txt = Text(root)
txt.place(rely=0.7, relheight=0.4)
txt.insert(END, 'hxeli dchsiadf s \n')
txt.insert(END, 'hxeli dchsVDFDSVBDFiadf s')

root.mainloop()