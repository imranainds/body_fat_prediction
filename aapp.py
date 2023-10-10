from flask import Flask

app = Flask(__name__)

from flask import Flask, render_template, request


import pickle
# Open the file in binary mode
with open(r'C:\Users\MD IMRAN\Desktop\NEW  ML PROJECT\neeeeeew\ml proo\bbbbbb\bodyfatmodell.pkl', 'rb') as file1:
    # Load the model from the file
    rf = pickle.load(file1)
# Close the file
file1.close()
app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        my_dict = request.form
        print(my_dict)
        
        density = float(my_dict['density'])
        abdomen = float(my_dict['abdomen'])
        chest = float(my_dict['chest'])
        weight = float(my_dict['weight'])
        hip = float(my_dict['hip'])
        input_features = [[density, abdomen, chest, weight, hip]]
        prediction = rf.predict(input_features)[0].round(2)
        # <p class="big-font">Hello World !!</p>', unsafe_allow_html=True
        string = 'Percentage of Body Fat Estimated is : ' + str(prediction)+'%'
        return render_template('show.html', string=string)
    return render_template('home.html')
if __name__ == "__main__":
    app.run(debug=True)