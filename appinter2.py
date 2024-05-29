import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
import seaborn as sns
from joblib import load
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from PIL import Image, ImageTk

# Define dataset and model paths
dataset_paths = {
    "Kidney Dataset": r"C:\Users\pc\Downloads\kidney (3).csv",
    "New Dataset": r"C:\Users\pc\Downloads\new_model.csv",
}

model_paths = {
    "Kidney Dataset": "C:\\Users\\pc\\Documents\\Python Scripts\\Models\\AI_ModelKedney1.joblib",
    "New Dataset": "C:\\Users\\pc\\Documents\\Python Scripts\\Models\\AI_model11.joblib",
}

welcome_screen = tk.Tk()
welcome_screen.title("Welcome to Heart Disease Predictor")

# Add a title at the top of the interface
title_label = tk.Label(welcome_screen, text="Heart Disease Predictor", font=("Helvetica", 24, "bold"))
title_label.pack(pady=20)

# Add an image to show the subject of the project
# Replace "heart.jpg" with the path to your image file
pil_image = Image.open("heart.jpg")

# Resize the image to fit the screen while maintaining aspect ratio
width, height = pil_image.size
max_height = welcome_screen.winfo_screenheight() - 100  # Adjust this value as needed
resize_factor = 0.5
new_width = int(width * resize_factor)
new_height = int(height * resize_factor)
pil_image = pil_image.resize((new_width, new_height))

# Convert the Pillow image to a Tkinter-compatible format
tk_image = ImageTk.PhotoImage(pil_image)
image_label = tk.Label(welcome_screen, image=tk_image)
image_label.pack(pady=20)

# Add a list of student names
student_names = "Presented By:Benmeridja Ahmed Younes"
student_label = tk.Label(welcome_screen, text=student_names, font=("Helvetica", 12))
student_label.pack()

# Run the Tkinter event loop for the welcome screen
welcome_screen.mainloop()

root = tk.Tk()
root.title("Machine Learning Model Visualization")

selected_dataset = tk.StringVar()
selected_model = tk.StringVar()

def dataset_selection_changed(event):
    selected_dataset_name = dataset_combobox.get()

def model_selection_changed(event):
    selected_model.set(model_combobox.get())

# Create dataset selection dropdown menu
dataset_label = tk.Label(root, text="Select Dataset:")
dataset_label.pack(side=tk.TOP, padx=10)

dataset_combobox = ttk.Combobox(root, textvariable=selected_dataset)
dataset_combobox['values'] = tuple(dataset_paths.keys())
dataset_combobox.pack(side=tk.TOP)
dataset_combobox.bind("<<ComboboxSelected>>", dataset_selection_changed)

# Create model path selection dropdown menu
model_label = tk.Label(root, text="Select Model Path:")
model_label.pack(side=tk.TOP, padx=10)

model_combobox = ttk.Combobox(root, textvariable=selected_model)
model_combobox['values'] = tuple(model_paths.keys())
model_combobox.pack(side=tk.TOP)
model_combobox.bind("<<ComboboxSelected>>", model_selection_changed)

def display_histogram():
    # Load the selected dataset
    dataset_path = dataset_paths.get(selected_dataset.get())
    df = pd.read_csv(dataset_path)

    # Create histogram plot
    plt.figure(figsize=(15, 10))
    df.drop('Class', axis=1).hist(bins=80, color="red")
    plt.tight_layout()

    # Display plot in popup window
    popup = tk.Toplevel(root)
    popup.title("Histogram Plot")
    canvas = FigureCanvasTkAgg(plt.gcf(), master=popup)
    canvas.draw()
    canvas.get_tk_widget().pack()

def display_boxplot():
    # Load the selected dataset
    dataset_path = dataset_paths.get(selected_dataset.get())
    df = pd.read_csv(dataset_path)

    # Create a scrollable frame within the popup window
    def on_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    popup = tk.Toplevel(root)
    popup.title("Boxplot Images")

    canvas = tk.Canvas(popup)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(popup, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind("<Configure>", on_configure)

    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor=tk.NW)

    # Create boxplot for each variable based on the selected dataset
    for i, column in enumerate(df.columns):
        if column != "Class":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df, x='Class', y=column, palette="Set2", ax=ax)
            ax.set_title(f'Boxplot of "{column}" based on the class')
            ax.set_xlabel('Class (Sick / Not sick)')
            ax.set_ylabel(column)

            # Display plot in frame
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack()

def display_heatmap():
    # Load the selected dataset
    dataset_path = dataset_paths.get(selected_dataset.get())
    df = pd.read_csv(dataset_path)

    # Create a scrollable frame within the popup window
    def on_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    popup = tk.Toplevel(root)
    popup.title("Heatmap")

    canvas = tk.Canvas(popup)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(popup, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind("<Configure>", on_configure)

    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor=tk.NW)

    # Create heatmap plot based on the selected dataset
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr().round(1), annot=True)
    plt.title("Correlation Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Features")

    # Display plot in frame
    canvas = FigureCanvasTkAgg(plt.gcf(), master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

def display_tsne_plot():
    # Load the selected dataset
    dataset_path = dataset_paths.get(selected_dataset.get())
    df = pd.read_csv(dataset_path)

    # Create a scrollable frame within the popup window
    def on_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    popup = tk.Toplevel(root)
    popup.title("t-SNE Plot")

    canvas = tk.Canvas(popup)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(popup, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind("<Configure>", on_configure)

    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor=tk.NW)

    # Create t-SNE plot based on the selected dataset
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(df.drop(columns=['Class']))

    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['Class'], cmap='viridis')
    plt.title('t-SNE Plot')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # Display plot in frame
    canvas = FigureCanvasTkAgg(plt.gcf(), master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

def display_class_distribution():
    # Load the selected dataset
    dataset_path = dataset_paths.get(selected_dataset.get())
    df = pd.read_csv(dataset_path)

    # Create a popup window
    popup = tk.Toplevel(root)
    popup.title("Distribution of Classes")

    # Create subplots for class distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(df['Class'], ax=ax)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")

    # Show the plot in the popup window
    canvas = FigureCanvasTkAgg(fig, master=popup)
    canvas.draw()
    canvas.get_tk_widget().pack()



def display_confusion_matrix():
    # Load the selected dataset
    dataset_path = dataset_paths.get(selected_dataset.get())
    df = pd.read_csv(dataset_path)
    df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

    # Load the corresponding model based on the selected model
    model_path = model_paths.get(selected_model.get())
    model = load(model_path)

    # Prepare data
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Make predictions using the loaded model
    y_pred = model.predict(X)

    # Preprocess class labels
    class_to_drop = 'Healthy'  # Adjust as needed
    unique_classes = df['Class'].astype(str).str.strip().unique()  # Convert to string and remove leading/trailing spaces
    print("Unique class labels in the dataset:", unique_classes)

    if class_to_drop not in unique_classes:
        raise ValueError(f"'{class_to_drop}' is not a valid class label.")

    # Drop the desired class for evaluation
    y = y[y != class_to_drop]
    y_pred = y_pred[y != class_to_drop]

    # Generate confusion matrix
    cm = confusion_matrix(y, y_pred)

    # Display confusion matrix using ConfusionMatrixDisplay
    popup = tk.Toplevel(root)
    popup.title("Confusion Matrix")

    # Create ConfusionMatrixDisplay object
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['CKD', 'Healthy'])

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_display.plot(ax=ax, cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    # Display plot in popup window
    canvas = FigureCanvasTkAgg(fig, master=popup)
    canvas.draw()
    canvas.get_tk_widget().pack()


def display_roc_curve():
    # Load the selected dataset
    dataset_path = dataset_paths.get(selected_dataset.get())
    df = pd.read_csv(dataset_path)

    # Load the corresponding model based on the selected model
    model_path = model_paths.get(selected_model.get())
    model = load(model_path)

    # Prepare data
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Remove 'Unnamed: 0' column from data if present
    X_cleaned = X.drop(columns=['Unnamed: 0'], errors='ignore')

    # Make predictions using the loaded model
    y_pred_proba = model.predict_proba(X_cleaned)[:, 1]

    # Calculate fpr, tpr, and thresholds for the ROC curve
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)

    # Calculate the area under the ROC curve (AUC)
    roc_auc = roc_auc_score(y, y_pred_proba)

    # Determine the image path based on the selected model
    if model_path == "C:\\Users\\pc\\Documents\\Python Scripts\\Models\\AI_ModelKedney1.joblib":
        image_path = r"C:\Users\pc\Desktop\courbe_roc.png"
    elif model_path == "C:\\Users\\pc\\Documents\\Python Scripts\\Models\\AI_model11.joblib":
        image_path = r"C:\Users\pc\Desktop\courbe_rocNewModel.png"
    else:
        # Default to a generic image if the model is not recognized
        image_path = r"C:\Users\pc\Desktop\ai.png"

    # Load the image and display it
    try:
        pil_image = Image.open(image_path)
        tk_image = ImageTk.PhotoImage(pil_image)

        # Create a popup window
        popup = tk.Toplevel(root)
        popup.title("ROC Curve Image")

        # Add the image to the popup window
        image_label = tk.Label(popup, image=tk_image)
        image_label.image = tk_image  # Keep a reference to avoid garbage collection
        image_label.pack()

    except Exception as e:
        print("Error loading image:", e)
def display_roc_curve():
    # Load the selected dataset
    dataset_path = dataset_paths.get(selected_dataset.get())
    df = pd.read_csv(dataset_path)

    # Load the corresponding model based on the selected dataset
    model_path = model_paths.get(selected_model.get())
    model = load(model_path)

    # Split the dataset into features (X) and target labels (y)
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)

    # Remove 'Unnamed: 0' column from test data if present
    X_test_cleaned = X_test.drop(columns=['Unnamed: 0'], errors='ignore')

    # Make predictions on the cleaned test data
    y_pred = model.predict(X_test_cleaned)

    # Calculate fpr, tpr, and thresholds for the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # Calculate the area under the ROC curve (AUC)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


# Create buttons
button_width = 50
button_height = 5

histogram_button = tk.Button(root, text="Histogram", command=display_histogram, width=button_width, height=button_height)
histogram_button.pack()

boxplot_button = tk.Button(root, text="Boxplot", command=display_boxplot, width=button_width, height=button_height)
boxplot_button.pack()

heatmap_button = tk.Button(root, text="Heatmap", command=display_heatmap, width=button_width, height=button_height)
heatmap_button.pack()

tsne_button = tk.Button(root, text="t-SNE Plot", command=display_tsne_plot, width=button_width, height=button_height)
tsne_button.pack()

class_distribution_button = tk.Button(root, text="Class Distribution", command=display_class_distribution, width=button_width, height=button_height)
class_distribution_button.pack()

confusion_matrix_button = tk.Button(root, text="Confusion Matrix", command=display_confusion_matrix, width=button_width, height=button_height)
confusion_matrix_button.pack()

roc_curve_button = tk.Button(root, text="ROC Curve", command=display_roc_curve, width=button_width, height=button_height)
roc_curve_button.pack()
root.mainloop(), 
