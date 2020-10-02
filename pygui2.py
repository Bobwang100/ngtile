from tkinter import *
# #定义一个顶级大窗口
# root = Tk()
# #在大窗口下定义一个菜单实例
# menubar = Menu(root)
# root.geometry('920x480')
# root.title('轮毂型号识别系统')
# #给菜单实例增加菜单项
#
# # 在大窗口下定义一个顶级菜单实例
# menubar = Menu(root)
#
#
# def func1():
#     print('hi')
# # 在顶级菜单实例下创建子菜单实例
# fmenu = Menu(menubar)
# for each in ['新建', '打开', '保存', '另存为']:
#     fmenu.add_command(label=each, command=func1)
#
# vmenu = Menu(menubar)
# # 为每个子菜单实例添加菜单项
# for each in ['复制', '粘贴', '剪切']:
#     vmenu.add_command(label=each)
#
# emenu = Menu(menubar)
# for each in ['默认视图', '新式视图']:
#     emenu.add_command(label=each)
#
# amenu = Menu(menubar)
# for each in ['版权信息', '联系我们']:
#     amenu.add_command(label=each)
#
# # 为顶级菜单实例添加菜单，并级联相应的子菜单实例
# menubar.add_cascade(label='文件', menu=fmenu, font=('宋体', 12))
# menubar.add_cascade(label='视图', menu=vmenu, font=('宋体', 12))
# menubar.add_cascade(label='编辑', menu=emenu)
# menubar.add_cascade(label='关于', menu=amenu)
#
# # 顶级菜单实例应用到大窗口中
# root['menu'] = menubar
# root.mainloop()

import tkinter as tk


class basedesk():
    def __init__(self, master):
        self.root = master
        self.root.config()
        self.root.title('Base page')
        self.root.geometry('920x480')

        initface(self.root)


class initface():
    def __init__(self, master):
        self.master = master
        # self.master.config(bg='green')
        # 基准界面initface
        self.initface = tk.Frame(self.master, )
        self.initface.pack()
        btn = tk.Button(self.initface, text='change', command=self.change)
        btn.pack()

    def change(self, ):
        self.initface.destroy()
        face1(self.master)


class face1():
    def __init__(self, master):
        self.master = master
        # self.master.config(bg='blue')
        self.face1 = tk.Frame(self.master, )
        self.face1.pack()

        # 在大窗口下定义一个顶级菜单实例
        menubar = Menu(root)

        # 在顶级菜单实例下创建子菜单实例
        fmenu = Menu(menubar)
        for each in ['新建', '打开', '保存', '另存为']:
            fmenu.add_command(label=each)

        vmenu = Menu(menubar)
        # 为每个子菜单实例添加菜单项
        for each in ['复制', '粘贴', '剪切']:
            vmenu.add_command(label=each)

        emenu = Menu(menubar)
        for each in ['默认视图', '新式视图']:
            emenu.add_command(label=each)

        amenu = Menu(menubar)
        for each in ['版权信息', '联系我们']:
            amenu.add_command(label=each)

        # 为顶级菜单实例添加菜单，并级联相应的子菜单实例
        menubar.add_cascade(label='文件', menu=fmenu, font=('宋体', 12))
        menubar.add_cascade(label='视图', menu=vmenu, font=('宋体', 12))
        menubar.add_cascade(label='编辑', menu=emenu)
        menubar.add_cascade(label='关于', menu=amenu)

        # 顶级菜单实例应用到大窗口中
        root['menu'] = menubar

        # lb1 = Label(root, text='D 模型生成', bg='#d3fbfb', font=('华文新魏', 14), relief=SUNKEN)
        # lb1.place(relx=0.8, rely=0.05, relwidth=0.15, relheight=0.1)


        btn_back = tk.Button(self.face1, text='face1 back', command=self.back, bg='red')
        btn_back.pack()

        # btn_back.place(relx=0.5, rely=0.35, relwidth=0.1, relheight=0.1)
    def back(self):
        self.face1.destroy()
        initface(self.master)


if __name__ == '__main__':
    root = tk.Tk()
    basedesk(root)
    root.mainloop()