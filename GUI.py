import tkinter as tk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from tkinter import *
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import numpy as np
import pandas as pd
import os
import seaborn as sns
import datetime as dt
import plotly.express as px
import sklearn
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.mixture import GaussianMixture as GMM
import itertools
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image, ImageTk
import cv2
import plotly.io as pio
import kaleido

from project import Project


class Machinelearning:
    def __init__(self):
        self.window = tk.Tk()
        self.window.geometry("1280x720")

        self.window.title("Machine Learning (Kelas B/Kelompok 8)")
        self.pr = Project()

        self.data = self.pr.ori_data()
        self.data_uk = self.pr.uk_data()

        self.frame_botright = ''
        self.panel_botright = ''
        self.tv = ''

        self.initial = 1

        self.process()

        self.main_frame()
        self.window.mainloop()

    def main_frame(self):
        # top_left = tk.Frame(master=self.window, bd=1, width=180, height=420, relief="raised")
        frame_left = tk.Frame(master=self.window, width=300, height=720)
        frame_left.pack(fill=tk.BOTH, side=tk.LEFT, padx=5, pady=5)

        top_left = tk.LabelFrame(master=frame_left, width=250, height=300,
                                 text='Menu')
        top_left.pack(fill=tk.BOTH, side=tk.TOP)

        self.b_dataset = tk.Button(top_left, text="Data Set", width=15, command=self.table)
        self.b_dataset.grid(row=0, column=0, padx=10, pady=10)

        # self.b_dataclean = tk.Button(top_left, text="Data Clean", width=15, command=self.table_clean)
        # self.b_dataclean.grid(row=1, column=0, padx=10, pady=10)

        self.b_RFM = tk.Button(top_left, text="Graphics", width=15, command=self.frame_RFM)
        self.b_RFM.grid(row=2, column=0, padx=10, pady=10)

        self.b_GMM = tk.Button(top_left, text="Analyze", width=15)
        self.b_GMM.grid(row=3, column=0, padx=10, pady=10)

        self.bot_left = tk.LabelFrame(master=frame_left, width=250, height=500,
                                      text='Info')
        self.bot_left.pack(fill=tk.BOTH, side=tk.TOP)

        # Frame Kanan
        self.frame_right = tk.LabelFrame(master=self.window, width=900, height=720,
                                         text='Figure')
        self.frame_right.pack(fill=tk.BOTH, side=tk.LEFT, expand=True, padx=5, pady=5)

        # frame kanan atas
        self.frame_topright = tk.Frame(master=self.frame_right, width=900, height=150)
        self.frame_topright.pack(fill=tk.BOTH, side=tk.TOP, padx=5, pady=5)
        # self.frame_topright.grid(row=0, column=0, padx=5, pady=0)

        # self.frame_botright.grid(row=1, column=0, padx=5, pady=0)

    def refresh(self):
        self.frame_right.destroy()
        self.frame_topright.destroy()
        self.frame_right = tk.LabelFrame(master=self.window, width=900, height=720, text='Figure')
        self.frame_right.pack(fill=tk.BOTH, side=tk.LEFT, expand=True, padx=5, pady=5)
        self.frame_topright = tk.Frame(master=self.frame_right, width=900, height=150)
        self.frame_topright.pack(fill=tk.BOTH, side=tk.TOP, padx=5, pady=5)

    def plot(self):
        # the figure that will contain the plot
        fig = Figure(figsize=(5, 5),
                     dpi=100)

        # list of squares
        y = [i ** 2 for i in range(101)]

        # adding the subplot
        plot1 = fig.add_subplot(111)

        # plotting the graph
        plot1.plot(y)

        # creating the Tkinter canvas
        # containing the Matplotlib figure
        canvas = FigureCanvasTkAgg(fig,
                                   master=self.top_right)
        canvas.draw()

        # placing the canvas on the Tkinter window
        canvas.get_tk_widget().pack()

        # creating the Matplotlib toolbar
        toolbar = NavigationToolbar2Tk(canvas,
                                       self.top_right)
        toolbar.update()

        # placing the toolbar on the Tkinter window
        canvas.get_tk_widget().pack()

    def plot2(self, figure):
        fig, ax = plt.subplots(figsize=(11, 9))
        # creating the Tkinter canvas
        # containing the Matplotlib figure
        canvas = plt.Figure()
        canvas = FigureCanvasTkAgg(fig,
                                   master=self.frame_botright)
        canvas.draw()

        # placing the canvas on the Tkinter window
        canvas.get_tk_widget().pack()

        # creating the Matplotlib toolbar
        toolbar = NavigationToolbar2Tk(canvas,
                                       self.frame_botright)
        toolbar.update()

        # placing the toolbar on the Tkinter window
        canvas.get_tk_widget().pack()

    def info(self, origin):
        flData = tk.Frame(master=self.bot_left, width=250, height=20)
        flData.grid(row=0, column=0, padx=5, pady=0)
        labelData = tk.Label(master=flData, text="Data \t= ")
        labelData.grid(row=0, column=0)
        labelData2 = tk.Label(master=flData, text=origin)
        labelData2.grid(row=0, column=1)

        if origin == "Original":
            row = str(len(self.data))
        elif origin == "Cleaning":
            row = str(len(self.data_uk))

        flRows = tk.Frame(master=self.bot_left, width=250, height=20)
        flRows.grid(row=1, column=0, padx=5, pady=0)
        labelData = tk.Label(master=flRows, text="Rows \t= ")
        labelData.grid(row=0, column=0)
        labelData2 = tk.Label(master=flRows, text=row)
        labelData2.grid(row=0, column=1)

    def table(self):
        self.refresh()
        if self.panel_botright == True:
            self.panel_botright.destroy()

        # frame kanan bawah
        self.frame_botright = tk.Frame(master=self.frame_right, width=900, height=500)
        self.frame_botright.pack(fill=tk.BOTH, side=tk.TOP, expand=True, padx=5, pady=5)

        self.tv = ttk.Treeview(master=self.frame_botright)

        tvscrolly = tk.Scrollbar(master=self.frame_botright, orient="vertical", command=self.tv.yview)
        tvscrollx = tk.Scrollbar(master=self.frame_botright, orient="horizontal", command=self.tv.xview)

        self.tv.configure(xscrollcommand=tvscrollx.set, yscrollcommand=tvscrolly.set)
        tvscrollx.pack(side="bottom", fill="x")
        tvscrolly.pack(side="right", fill="y")

        self.tv.pack(fill=tk.BOTH, side=tk.BOTTOM, expand=True)

        self.tv.delete(*self.tv.get_children())

        def load_ori():
            self.load_data(self.data)
            self.info("Original")

        def load_clean():
            self.load_data(self.data_uk)
            self.info("Cleaning")

        b_original = tk.Button(self.frame_topright, text="Data Original", width=15, command=load_ori)
        b_original.grid(row=0, column=0, padx=10, pady=5)

        b_clean = tk.Button(self.frame_topright, text="Data Clean", width=15, command=load_clean)
        b_clean.grid(row=0, column=1, padx=10, pady=5)

    def load_data(self, data):
        self.tv["column"] = list(data.columns)
        self.tv["show"] = "headings"
        for column in self.tv["columns"]:
            self.tv.heading(column, text=column)

        df_rows = data.to_numpy().tolist()
        for row in df_rows:
            self.tv.insert("", "end", values=row)
        return None

    def process(self):
        # heatmap
        correlation = self.data_uk.corr(method="kendall")
        sns.heatmap(correlation, vmin=-1, vmax=1, annot=True)
        # heatmap = py.heat_2d(correlation, master=self.frame_botright)
        plt.savefig('figure/heatmap.jpg')

        # monetary recency frekuenci
        data_mo = self.pr.monetary()
        data_fr = self.pr.frequency()
        data_re = self.pr.recency()
        self.RFM = self.pr.rfm(data_re, data_fr, data_mo)

        self.box_rfm("Recency")
        self.box_rfm("Frequency")
        self.box_rfm("Amount")

        #BIC
        self.pr.bic_score(self.RFM)

        #AIC
        self.pr.bic_aic(self.RFM)


    def load_image(self, filename):
        matrix_img_hist = cv2.imread(filename)
        matrix_img_hist = cv2.resize(matrix_img_hist, dsize=(850, 450), interpolation=cv2.INTER_CUBIC)
        matrix_img_hist = cv2.cvtColor(matrix_img_hist, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(image=Image.fromarray(matrix_img_hist))

        self.panel_botright.configure(image=img)
        self.panel_botright.image = img

    def box_rfm(self, data):

        # outlier treatment: We will delete everything outside the IQR

        Q1 = self.RFM.Amount.quantile(0.25)
        Q3 = self.RFM.Amount.quantile(0.75)
        IQR = Q3 - Q1
        RFM = self.RFM[(self.RFM.Amount >= (Q1 - 1.5 * IQR)) & (self.RFM.Amount <= (Q1 + 1.5 * IQR))]

        # outlier treatment : We will delete everything outside the IQR

        Q1 = RFM.Frequency.quantile(0.25)
        Q3 = RFM.Frequency.quantile(0.75)
        IQR = Q3 - Q1
        RFM = RFM[(RFM.Frequency >= (Q1 - 1.5 * IQR)) & (RFM.Frequency <= (Q1 + 1.5 * IQR))]

        fig = px.box(RFM, y=data)
        if data == "Recency":
            file = 'figure/Recency.png'
        elif data == "Frequency":
            file = 'figure/Frequency.png'
        elif data == "Amount":
            file = 'figure/Monetary.png'
        # fig.show()
        pio.write_image(fig, file, format='png')
        # pio.write_image(fig, save_fig, format='jpg')
        # plt.savefig(save_fig, fig)
        # cv2.imwrite(save_fig, fig)


    def frame_RFM(self):
        self.refresh()
        if self.tv == True:
            self.tv.destroy()

        if self.frame_botright == True:
            self.frame_botright.destroy()

        def button_re():
            self.load_image("figure/Recency.png")

        def button_fr():
            self.load_image("figure/Frequency.png")

        def button_mo():
            self.load_image("figure/Monetary.png")

        def button_ht():
            self.load_image("figure/heatmap.jpg")

        def button_bic():
            self.load_image("figure/bic_score.png")

        def button_aic():
            self.load_image("figure/aic_score.png")

        self.panel_botright = tk.Label(master=self.frame_right, width=900, height=500)
        self.panel_botright.pack()

        b_heatmap = tk.Button(self.frame_topright, text="Heatmap", width=15, command=button_ht)
        b_heatmap.grid(row=0, column=0, padx=10, pady=5)

        b_recency = tk.Button(self.frame_topright, text="Recency", width=15, command=button_re)
        b_recency.grid(row=0, column=1, padx=10, pady=5)

        b_frequency = tk.Button(self.frame_topright, text="Frequency", width=15, command=button_fr)
        b_frequency.grid(row=0, column=2, padx=10, pady=5)

        b_monetary = tk.Button(self.frame_topright, text="Monetary", width=15, command=button_mo)
        b_monetary.grid(row=0, column=3, padx=10, pady=5)

        b_bic = tk.Button(self.frame_topright, text="BIC Score", width=15, command=button_bic)
        b_bic.grid(row=0, column=4, padx=10, pady=5)

        b_aic = tk.Button(self.frame_topright, text="AIC Score", width=15, command=button_aic)
        b_aic.grid(row=0, column=5, padx=10, pady=5)


Machinelearning()
