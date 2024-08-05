

---

# Final AI Project: Sentiment Analysis with Naive Bayes Classification

This project demonstrates a sentiment analysis application using the Naive Bayes classification method. The project incorporates web technologies such as HTML and Tailwind CSS, and leverages Python (Flask) for the backend. Additionally, it uses Node.js for managing dependencies and building CSS.

## Tools and Technologies

- HTML
- Tailwind CSS
- Python (Flask)
- npm, Node.js

## Getting Started

Follow the steps below to set up and run the project on your local machine.

### 1. Set Up Python Environment

First, create a virtual environment and activate it:

```bash
python3 -m venv Env
source Env/bin/activate
```

### 2. Install Python Libraries

Install the required Python libraries using pip:

```bash
pip install numpy seaborn joblib scikit-learn matplotlib pandas flask
```

### 3. Install Node.js Dependencies

Make sure you have Node.js installed on your computer. Then, install the necessary Node.js dependencies:

```bash
npm install
npm install -D tailwindcss
npx tailwindcss init
npm install flowbite
```

### 4. Run Tailwind CSS

To build the CSS using Tailwind, run:

```bash
npm run css-style
```

### 5. Run Python Server

Open a new terminal window and start the Flask server:

```bash
python3 app.py runserver
```

The server will run on [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Directory Structure

```
.
├── Env/
├── app.py
├── static/
│   ├── css/
│   └── js/
├── templates/
│   └── index.html
├── package.json
├── tailwind.config.js
└── README.md
```

## Contributing

Feel free to contribute to this project by submitting a pull request. Please ensure all changes are well-documented and tested.

## Contact

For any inquiries, please contact [Me](mailto:heykalsayid@gmail.com).

---
Have fun!
