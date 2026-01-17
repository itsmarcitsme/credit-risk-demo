import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="Credit Risk — Default Probability", layout="wide")

@st.cache_data
def simulate(n=4000, seed=42):
    rng=np.random.default_rng(seed)
    age=rng.integers(18,75,n)
    income=rng.normal(32000,14000,n).clip(6000,180000)
    loan_amount=rng.normal(18000,12000,n).clip(500,120000)
    loan_term=rng.choice([12,24,36,48,60,72],n,p=[0.12,0.18,0.25,0.18,0.20,0.07])
    rate=rng.normal(7.5,3.0,n).clip(1.5,22)
    employment=rng.choice(["Salaried","Self-employed","Public servant","Unemployed"],n,p=[0.56,0.25,0.12,0.07])
    province=rng.choice(["Girona","Barcelona","Tarragona","Lleida","Other"],n,p=[0.18,0.40,0.12,0.08,0.22])
    delinq=np.clip(rng.poisson(0.25,n),0,5)
    dti=np.clip((loan_amount/(income+1))*rng.normal(0.65,0.15,n),0.02,2.5)
    credit_score=rng.normal(660,80,n).clip(300,850)

    logit=(-4.2+0.012*(loan_amount/1000)+1.1*np.maximum(dti-0.45,0)+0.35*delinq
           -0.004*(credit_score-650)-0.00001*(income-30000)
           +np.where(employment=="Unemployed",1.2,0)
           +np.where(employment=="Self-employed",0.25,0)
           +np.where(loan_term>=72,0.3,0))
    prob=1/(1+np.exp(-logit))
    y=rng.binomial(1,prob)

    df=pd.DataFrame({
        "age":age,
        "annual_income":income.round(0),
        "loan_amount":loan_amount.round(0),
        "term_months":loan_term,
        "interest_rate":rate.round(2),
        "employment":employment,
        "province":province,
        "prior_delinquencies":delinq,
        "dti":dti.round(3),
        "credit_score":credit_score.round(0),
        "default":y
    })
    # missingness
    for col in ["annual_income","credit_score","employment"]:
        df.loc[rng.choice(n,int(0.03*n),replace=False),col]=np.nan
    return df

@st.cache_resource
def train_model(df: pd.DataFrame):
    X=df.drop(columns=["default"])
    y=df["default"]

    num=["age","annual_income","loan_amount","term_months","interest_rate","prior_delinquencies","dti","credit_score"]
    cat=["employment","province"]

    pre=ColumnTransformer([
        ("num",Pipeline([("imp",SimpleImputer(strategy="median"))]),num),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                         ("ohe",OneHotEncoder(handle_unknown="ignore"))]),cat)
    ])

    model=Pipeline([("pre",pre),("clf",LogisticRegression(max_iter=800))])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    model.fit(Xtr,ytr)

    auc=roc_auc_score(yte, model.predict_proba(Xte)[:,1])
    return model, float(auc)

st.title("Credit Risk — Default Probability (Demo)")
st.caption("Demo portfolio: model trained on synthetic data (replaceable with real data).")

df = simulate()
model, auc = train_model(df)

colA, colB = st.columns([1,1])

with colA:
    st.subheader("Applicant inputs")
    age = st.slider("Age", 18, 75, 34)
    income = st.number_input("Annual income", min_value=0, max_value=300000, value=42000, step=1000)
    loan_amount = st.number_input("Loan amount", min_value=0, max_value=200000, value=15000, step=500)
    term = st.selectbox("Term (months)", [12,24,36,48,60,72], index=3)
    rate = st.slider("Interest rate (%)", 1.5, 22.0, 7.2)
    delinq = st.slider("Prior delinquencies", 0, 5, 0)
    dti = st.slider("DTI", 0.0, 2.5, 0.29, step=0.01)
    credit_score = st.slider("Credit score", 300, 850, 710)
    employment = st.selectbox("Employment", ["Salaried","Self-employed","Public servant","Unemployed"], index=0)
    province = st.selectbox("Province", ["Girona","Barcelona","Tarragona","Lleida","Other"], index=1)

    row = pd.DataFrame([{
        "age": age,
        "annual_income": income,
        "loan_amount": loan_amount,
        "term_months": term,
        "interest_rate": rate,
        "prior_delinquencies": delinq,
        "dti": dti,
        "credit_score": credit_score,
        "employment": employment,
        "province": province
    }])

    if st.button("Predict default probability", type="primary"):
        p = model.predict_proba(row)[:,1][0]
        st.metric("Default probability", f"{p:.2%}")
        st.write("Model quality (ROC-AUC on holdout):", f"**{auc:.3f}**")

with colB:
    st.subheader("Data preview (synthetic)")
    st.dataframe(df.sample(20, random_state=1), height=420)
    st.write("Tip: in a real project, replace the synthetic generator with a real dataset and keep the same pipeline.")
