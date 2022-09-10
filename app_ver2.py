#import文
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from sympy import rotations

#事前準備
st.set_page_config(page_title='統計体験教室',layout='wide')
sns.set(font='IPAexGothic')

#MENU
menu = st.sidebar.radio(label='MENU',options=['TOP','講義','体験','(おまけ)LaTeX試し書き'])


#TOP
if menu == 'TOP':
    st.title('Welcome!!')
    st.write('画面左部のMENUからスタート')
    st.balloons()

#講義
elif menu == '講義':
    st.subheader('講義＞　「特徴量」をざっと学習')
    def l(str):
        st.latex(str)
    with st.expander(label='特徴量とは',expanded=False):
        l(r'''統計学における特徴量とは、データを変形して得られ、\\
            その特徴を表現し、続く処理に利用される数値である。''')
        l(r'''以下のタブから主要な特徴量の解説を見ることができる。''')
    with st.expander('最小値・最大値'):
        # l(r'''\textcolor{red}{point:一番小さなデータと大きなデータ。最も単純な特徴量。}''')
        l(r'''X = [x_1 , x_2 , \cdots , x_n]''')
        l(r'''このような一次元データXに対して、最小値と最大値は''')
        l(r'''min = min X = min_{i=1}^{n} x_i\\
            max = max X = max_{i=1}^{n} x_i''')
        l(r'''により求められる。''')
    with st.expander('四分位数'):
        # l(r'''\textcolor{red}{point:箱ひげ図として視覚化されることが多い特徴量。}''')
        l(r'''X = [x_1 , x_2 , \cdots , x_n]''')
        l(r'''このような一次元データXを昇順に並べ替え、次のように表すこととする。''')
        l(r'''[x_{i_1} , x_{i_1} , \cdots , x_{i_1}]''')
        l(r'''並べ替えた後のデータについて第一、第二、第三四分位数は、''')
        l(r'''Q_1 = x_{i_l} ~~,~~ Q_2 = x_{i_m} ~~,~~ Q_3 = x_{i_n}''')
        l(r'''の各値である。''')
        l(r'''但し、l,m,nはそれぞれ \frac{1}{4} n , \frac{2}{4} n , \frac{3}{4} n のように定める。\\
            l,m,nの値が整数とならないときは、前後の値の平均をとる。''')
        l(r'''例:[x_1,x_2,\cdots,x_{10}]が与えられた時、\\
            Q_1 = x_{i_3} ~~,~~ Q_2 = \frac{ x_{i_5} + x_{i_6} }{2} ~~,~~ Q_3 = x_{i_8}''')
    with st.expander('期待値（平均）'):
        # l(r'''\textcolor{red}{point:いわゆる平均のことを、統計学では期待値と呼ぶことが多い。}''')
        l(r'''X = [x_1 , x_2 , \cdots , x_n]''')
        l(r'''このような一次元データXに対して、期待値（平均）は''')
        l(r'''\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i''')
        l(r'''により求められる。''')
    with st.expander('分散・標準偏差'):
        l(r'''X = [x_1 , x_2 , \cdots , x_n]''')
        l(r'''このような一次元データXに対して、分散は''')
        l(r'''s_x^2 = \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2''')
        l(r'''即ち、偏差の2乗の平均で求められる。''')
        l(r'''また、標準偏差は分散の正の平方根を取ることで求められる。即ち、''')
        l(r'''s_x = \sqrt{s_x^2}''')
        l(r'''\textcolor{red}{point:分散・標準偏差はデータの「散らばり具合」を表す指標となる。}''')
    with st.expander('共分散・相関係数'):
        # l(r'''\textcolor{red}{point:共分散・相関係数の符号は相関を表す。}''')
        l(r'''X = [x_1 , x_2 , \cdots , x_n] \\
            Y = [y_1 , y_2 , \cdots , y_n]''') 
        l(r'''このような二次元データX,Yに対して、共分散は''')
        l(r'''s_{xy} = \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})''')
        l(r'''即ち、共分散は偏差積の平均で求められる。''')
        l(r'''また、相関係数は以下の式で求められる。''')
        l(r'''r = \frac{ s_{xy} }{s_x s_y}''')
    with st.expander('標準化・偏差値'):
        # l(r'''\textcolor{red}{point:データを標準化することにより異なるサイズのデータを比較しやすくなる。}''')
        l(r'''X = [x_1 , x_2 , \cdots , x_n]''')
        l(r'''このような一次元データXの標準化とは、各データx_iを''')
        l(r'''z_i = \frac{ x_i - \bar{x} }{s_x}''')
        l(r'''と変換することである。標準化したデータの期待値は0、標準偏差は1となる。''')
        l(r'''また、各データx_iの偏差値とは、''')
        l(r'''10 z_i + 50 = 10 \frac{ x_i - \bar{x} }{s_x} + 50''')
        l(r'''により得られる値である。\\
             各x_iについて偏差値をとったデータの期待値は50、標準偏差は10となる。''')

#体験
elif menu == '体験':
    with st.sidebar.expander('↓「体験」で使用するデータを準備'):
        if st.checkbox('ランダムにデータを作成',value=True):
            df = pd.DataFrame(
                data=np.random.randn(100,4)
            )  #標準正規分布に従う100行4列のデータ
        else:
            st.error('↓使用するデータをアップロード')
            file_path = st.file_uploader('*excel限定!',type='xlsx')
            if file_path:
                df = pd.read_excel(file_path)

    st.subheader('体験')
    col1,col2,col3 = st.columns(3)
    
    with st.expander('①データセットを表示'):
        st.dataframe(df,height=200)

    with st.expander('②特徴量分析'):
        col1,col2 = st.columns(2)
        S = col1.selectbox(
            label='選択',
            options=['最小値・最大値',
                    # '四分位数',
                    '期待値（平均）',
                    '分散・標準偏差',
                    '共分散・相関係数',
                    '標準化・偏差値']
        )
        df_n_cols = []
        for col in df.columns:
            if df[col].dtype == int or df[col].dtype == float:
                df_n_cols.append(col)

        if S == '最小値・最大値':
            col1.latex(r'''
                定義式\\
                min(x) = min_{i=1}^{n} x_i \\
                max(x) = max_{i=1}^{n} x_i
            ''')
            col2.write('計算結果')
            col2.table(pd.DataFrame(
                        data=[ [df[col].min(),df[col].max()] for col in df_n_cols ],
                        index=df_n_cols,
                        columns=['min','max']
                    ).T )
        elif S == '四分位数':
            pass
        elif S == '期待値（平均）':
            col1.latex(r'''
                定義式\\
                mean(x) = \bar{x} = \frac{1}{n} \sum_{i=1}^n x_i 
            ''')
            col2.write('計算結果')
            col2.table(pd.DataFrame(
                        data=[ df[col].mean() for i in df_n_cols ],
                        index=df_n_cols,
                        columns=['mean']
                    ).T )
        elif S == '分散・標準偏差':
            col1.latex(r'''
                定義式\\
                var(x) = s_x^2 = \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2 \\
                std(x) = s_x = \sqrt{s_x^2}
            ''')
            col2.write('計算結果')
            col2.table(pd.DataFrame(
                        data=[ [df[col].var(),df[col].std()] for i in df_n_cols ],
                        index=df_n_cols,
                        columns=['var','std']
                    ).T )
        elif S == '共分散・相関係数':
            col1.latex(r'''
                定義式\\
                cov(x,y) = s_{xy} = \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})\\
                cor(x,y) = r = \frac{ s_{xy} }{s_x s_y}
            ''')
            col2.write('計算結果')
            col2.caption('cov')
            col2.table(df.cov())
            col2.caption('cor')
            col2.table(df.corr())
        elif S == '標準化・偏差値':
            col1.latex(r'''
                定義式\\
                偏差値 = 10 \frac{ x_i - \bar{x} }{s_x} + 50
            ''')
            col2.write('計算結果')
            col2.caption('列ごとの偏差値')
            df_SS = pd.DataFrame(
                data=[ 
                    10*( df[col] - df[col].mean() ) / df[col].std() + 50 
                    for col in df_n_cols
                    ]
            ).T
            col2.dataframe(df_SS)
        else:
            pass

    with st.expander('③可視化'):
        col1,col2 = st.columns(2)
        plt_type = col1.selectbox(
            label='可視化方法',
            options=['箱ひげ図',
                    'ヒストグラム',
                    '散布図',
                    '折れ線グラフ']
        )
        df_n_cols = []
        for col in df.columns:
            if df[col].dtype == int or df[col].dtype == float:
                df_n_cols.append(col)
        if plt_type == '箱ひげ図':
            fig,ax = plt.subplots()
            ax.boxplot(df.loc[:,df_n_cols])
            ax.set_xticklabels(df_n_cols)
            col2.pyplot(fig)
        elif plt_type == 'ヒストグラム':
            col = col1.radio(label='選択',options=df.columns,horizontal=True)
            fig,ax = plt.subplots()
            if df[col].nunique() <= 30:
                ax.hist(df[col],bins=df[col].nunique())
            else:
                ax.hist(df[col],bins=30)
            plt.xticks(rotation=90)
            col2.pyplot(fig)
        elif plt_type == '散布図':
            X = col1.radio(label='X',options=df.columns,horizontal=True)
            Y = col1.radio(label='Y',options=df.columns,horizontal=True)
            fig,ax = plt.subplots()
            ax.scatter(x=df[X],y=df[Y])
            col2.pyplot(fig)
        elif plt_type == '折れ線グラフ':
            col = col1.radio(label='選択',options=df.columns,horizontal=True)
            fig,ax = plt.subplots()
            ax.plot(df[col])
            col2.pyplot(fig)
        else:
            pass

#おまけ
elif menu == '(おまけ)LaTeX試し書き':
    my_latex = st.text_area(label='LaTeX試し書き',value='\LaTeX',height=100)
    st.latex(my_latex)