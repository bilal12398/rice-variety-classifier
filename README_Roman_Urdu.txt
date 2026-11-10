==============================================================================
 RICE VARIETY CLASSIFIER  -  POORE PROJECT KI ROMAN URDU EXPLANATION
 Course: CT-361 Artificial Intelligence & Expert System (AIES)
 NED University of Engineering & Technology
 Group: Muhammad Bilal Uddin (CT-24036), Muhammad Usman (CT-24031),
        Muhammad Mustafa (CT-24038)
==============================================================================


1. PROJECT KYA HAI? (Short Summary)
------------------------------------------------------------------------------
Yeh ek Python based system hai jo chawal ke ek dane (grain) ki 7 numerical
measurements dekh kar bata deta hai ke woh Cammeo variety ka hai ya Osmancik
variety ka. Is mein data preprocessing, Decision Tree aur KNN models ki
training, unka comparison, hyper-parameter tuning, evaluation aur ek simple
interface (Streamlit) shamil hai.

Yeh project CCP (Complex Computing Problem) aur CCA (Complex Computing
Activity) dono documents ki requirements ke mutabiq banaya gaya hai.


2. PROBLEM STATEMENT
------------------------------------------------------------------------------
- Masla (Issue): Chawal ki alag alag varieties dekhne mein bohat milti julti
  hoti hain. Insaan se pehchan karwana slow hai aur har insaan ka andaza
  alag ho sakta hai.
- Real-world impact: Agar varieties mix ho jayen to quality kharab hoti hai,
  qeemat ka nuqsan hota hai aur supply chain mein dhoka (fraud) ho sakta hai.
  Automatic pehchan se mills mein quality control asaan ho jata hai.
- AI approach: Supervised Machine Learning (Classification). Hum Decision
  Tree aur K-Nearest Neighbours (KNN) use karte hain.

Input  : Dane ki 7 measurements.
Output : Variety - Cammeo ya Osmancik (saath mein confidence/probability).


3. DATASET
------------------------------------------------------------------------------
Dataset ka naam: Rice (Cammeo and Osmancik), UCI Machine Learning Repository.
Banane wale: Cinar aur Koklu (Selcuk University, Turkey), saal 2019.
Total rows: 3810 (Cammeo = 1630, Osmancik = 2180). Missing values: 0.
Duplicate rows: 0.

7 Features (columns):
  1. Area               - dane ke andar kitne pixels hain
  2. Perimeter          - dane ki boundary ki lambai
  3. Major_Axis_Length  - dane ki sab se lambi line (lambai)
  4. Minor_Axis_Length  - dane ki sab se chhoti line (chaurai)
  5. Eccentricity       - dane kitna gol ya lamba hai
  6. Convex_Area        - dane ko gher lene wale sab se chhote convex shape
                          ke pixels
  7. Extent             - dane ka area aur uske bounding box ka ratio
Target (Class): Cammeo ya Osmancik.

Note: Is project mein ek hi dataset use hua hai jo 3810 samples ke saath kaafi
bara aur munasib hai. CCA document is ko allow karta hai ("one sufficiently
large and suitable dataset").


4. PROJECT KI FILES
------------------------------------------------------------------------------
train_model.py                  -> poori training + evaluation ka code
app.py                          -> Streamlit interface
data/Rice_Cammeo_Osmancik.csv   -> dataset
models/rice_models.joblib       -> save kiye hue trained models
outputs/metrics.json            -> sab results (numbers)
outputs/results_table.csv       -> results ka table
outputs/figures/                -> sab graphs (PNG images)
report/Rice_Variety_Classification_Report.docx -> 16 page ki technical report
report/build_report.py          -> report dobara banane ka script
assets/ned_logo.png             -> report ke pehle page ka logo
requirements.txt                -> zaroori Python packages
README_Roman_Urdu.txt           -> yeh file


5. CODE KAISE KAAM KARTA HAI? (train_model.py - step by step)
------------------------------------------------------------------------------
Step 1: Data load
  CSV file pandas se load hoti hai. Shape, missing values, duplicates aur
  outliers check kiye jate hain (IQR rule se).
  Outliers kam hain (zyada tar Minor_Axis_Length mein 65 aur Eccentricity mein
  21). Inhein rakha gaya hai kyunke yeh asli dane hain, ghalti nahi.

Step 2: EDA (Exploratory Data Analysis)
  Class distribution, har feature ka histogram, correlation heatmap aur box
  plots banaye jate hain. Pata chalta hai ke Cammeo dane aam taur par bare aur
  lambe hote hain, Osmancik chhote aur gol. Lekin dono ka kuch hissa overlap
  karta hai, is liye 100% accuracy mumkin nahi.

Step 3: Preprocessing
  - Label encoding: Cammeo = 0, Osmancik = 1.
  - Stratified train/test split: 80% training (3048) aur 20% testing (762).
    "Stratified" ka matlab dono hisson mein class ka ratio same rehta hai.
    random_state = 42 hai taake har dafa same result aaye.
  - Median Imputer: agar koi value missing ho to median se bhar deta hai (is
    dataset mein missing nahi hai, magar pipeline robust rehti hai).
  - StandardScaler (sirf KNN ke liye): KNN distance par chalta hai. Area
    (~12000) aur Eccentricity (~0.9) ki scale bohat alag hai, is liye scaling
    zaroori hai. Decision Tree ko scaling ki zaroorat nahi.
  - Sab kuch scikit-learn Pipeline mein hai, is se data leakage nahi hoti.

Step 4: Baseline models (pehla muqabla)
  a) Majority Class (Dummy): hamesha Osmancik batata hai. Accuracy 57.22%.
     Yeh sirf yeh dikhane ke liye hai ke bina learning ke kitna score milta hai.
  b) Decision Tree (default settings).
  c) KNN (k = 5).

Step 5: Improvement - Hyper-parameter Tuning
  GridSearchCV se 5-fold stratified cross-validation ke zariye best settings
  dhoondi jati hain (sirf training data par, test data ko haath nahi lagaya).
  Decision Tree grid: 448 combinations, KNN grid: 64 combinations.
  Best settings:
    Decision Tree -> criterion = gini, max_depth = 4, min_samples_leaf = 10,
                     min_samples_split = 2, class_weight = balanced
    KNN           -> n_neighbors = 25, weights = uniform,
                     metric = euclidean

Step 6: Evaluation (unseen test set par)
  Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix aur
  5-fold CV accuracy nikali jati hai.

Step 7: Save
  Models joblib file mein, numbers metrics.json mein aur graphs figures
  folder mein save hote hain.


6. NATIJE (RESULTS - test set, 762 dane)
------------------------------------------------------------------------------
Model                      Accuracy   ROC-AUC   CV Accuracy
Majority Class (Dummy)     0.5722     0.5000    0.5722
Decision Tree (Baseline)   0.8976     0.8947    0.8868
KNN (Baseline)             0.9094     0.9490    0.9265
Decision Tree (Tuned)      0.9186     0.9715    0.9308
KNN (Tuned)                0.9094     0.9699    0.9308

Samajhne wali baatein:
 - Dono models dummy baseline se bohat behtar hain (57% se 90-92%).
 - Tuning se Decision Tree ki accuracy 89.76% se 91.86% ho gayi.
 - KNN ki test accuracy tuning ke baad wahi rahi (90.94%), lekin uska ROC-AUC
   0.949 se 0.970 aur CV accuracy 92.65% se 93.08% ho gayi. Matlab tuning ne
   KNN ko behtar aur stable banaya, magar is test split par sahi labels
   ziyada nahi barhe. Yeh baat honestly report mein bhi likhi gayi hai.
 - Test set sirf 762 dane ka hai, is liye 1% ka farq taqreeban 8 danon ka
   farq hai. Tuned Decision Tree aur tuned KNN ka farq chhota hai.
 - Sab se important feature Major_Axis_Length (importance 0.96) hai, yani
   dane ki lambai variety pehchanne mein sab se aham hai.


7. INTERFACE (app.py) KAISE KAAM KARTA HAI?
------------------------------------------------------------------------------
Streamlit browser mein ek web app kholta hai jis ke 4 tabs hain:
  1) Predict       : 7 sliders se measurements daalein. Predicted variety aur
                     confidence (%) dono models ke liye dikhai deti hai.
                     "Start from" option se Typical Cammeo / Typical
                     Osmancik / Dataset average ki values load ho jati hain.
  2) Model Results : baseline vs tuned ka table, best parameters, comparison
                     chart aur confusion matrices.
  3) Charts        : ROC curve, feature importance, KNN accuracy vs k,
                     Decision Tree depth curve aur tree ka top hissa.
  4) Dataset       : data ka preview, class ke hisaab se averages, histograms
                     aur correlation heatmap.


8. VS CODE MEIN CHALANE KA TARIQA (Step by Step)
------------------------------------------------------------------------------
Zaroorat: Python 3.10 ya us se naya (python.org se install karein, install
karte waqt "Add Python to PATH" par tick lagayein).

1) Zip file ko extract (unzip) karein. Folder "rice_variety_classifier" ban
   jayega.
2) VS Code kholein: File > Open Folder > rice_variety_classifier select karein.
3) VS Code mein Terminal kholein: Terminal > New Terminal.
4) (Optional magar behtar) Virtual environment:
       python -m venv venv
       venv\Scripts\activate          (Windows)
       source venv/bin/activate       (Mac / Linux)
5) Packages install karein:
       pip install -r requirements.txt
6) (Optional) Model dobara train karna ho to:
       python train_model.py
   Trained model pehle se zip mein maujood hai, isliye yeh step zaroori nahi.
   (Agar aap ke computer par scikit-learn ka version alag hua to app khud
   model dobara train kar legi, takreeban 1 minute lagega.)
   Is se results/graphs dobara ban jate hain (same results aayenge).
7) Interface chalayein:
       streamlit run app.py
   Browser mein khud khul jayega (aam taur par http://localhost:8501).
   Band karne ke liye terminal mein Ctrl + C dabayein.

Agar "python" command na chale to "py" ya "python3" try karein.
Agar "streamlit" command na chale to:  python -m streamlit run app.py

Dost ko share karna: poora zip file bhej dein. Woh bhi upar wale steps follow
kare. Dono ke computer par Python install hona chahiye.


9. REPORT MEIN LOGO LAGANA (ZAROORI)
------------------------------------------------------------------------------
Report ke pehle page par logo ki jagah ek placeholder (khali gol nishan)
hai, kyunke official NED logo mein yahan download nahi kar saka. Do tareeqe:
  Tareeqa 1 (asaan): Report ko Word mein kholein, placeholder picture par
     right-click > Change Picture > From a File, aur NED ka logo select karein.
  Tareeqa 2: Logo ko PNG bana kar assets/ned_logo.png ke naam se replace
     karein, phir terminal mein:
         pip install python-docx
         python report/build_report.py
     Report naye logo ke saath dobara ban jayegi.


10. VIVA / PRESENTATION KE LIYE ASAAN JAWABAT
------------------------------------------------------------------------------
Q: Decision Tree kya hai?
A: Yeh if-then rules ka tree hai. Har node par ek feature par threshold se
   data ko do hisson mein baanta jata hai jab tak class alag na ho jayen.

Q: KNN kya hai?
A: Naye dane ke sab se qareeb k training dane dekhta hai aur unki majority
   class de deta hai.

Q: KNN mein scaling kyun?
A: Kyunke KNN distance se kaam karta hai; bari scale wala feature (Area)
   distance par ghalib aa jata hai.

Q: Baseline kyun?
A: Taake pata chale ke hamare model ne asal mein kitna seekha. Dummy model
   57.22% deta hai, hamare models 90% se upar.

Q: Overfitting kya hai aur kaise roka?
A: Jab model training data ratta laga le magar naye data par kharab chale.
   Decision Tree ki depth aur leaf size ko cross-validation se limit kiya
   (max_depth = 4, min_samples_leaf = 10).

Q: Sirf accuracy kyun kaafi nahi?
A: Classes thori imbalanced hain (57% / 43%), is liye Precision, Recall, F1,
   ROC-AUC aur Confusion Matrix bhi dekhe.

Q: Ethics aur privacy?
A: Dataset mein sirf dane ki physical measurements hain, koi personal data
   nahi. Decision Tree transparent hai, har prediction ke saath confidence
   dikhai jati hai, aur final faisla insaan inspector ka rehta hai.

Q: Limitations?
A: Sirf 2 varieties; classes overlap karti hain; camera/light badalne par
   measurements badal sakti hain; sirf 762 test dane.

==============================================================================
 KHATAM
==============================================================================
