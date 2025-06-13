import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

class DataGenerator:
    def __init__(self, num_samples=10000):
        self.num_samples = num_samples
        self.imiona = ["Jan", "Anna", "Piotr", "Katarzyna", "Marek", "Maria", "Tomasz", "Agnieszka", "Paweł", "Joanna"]
        self.nazwiska = ["Kowalski", "Nowak", "Wiśniewski", "Wójcik", "Kamiński", "Lewandowski", "Zieliński", "Szymański", "Dąbrowski", "Woźniak"]
        self.miasta = ["Warszawa", "Kraków", "Gdańsk", "Wrocław", "Poznań", "Łódź", "Szczecin", "Lublin", "Katowice", "Bydgoszcz"]
        self.ocena = [1, 2, 3, 4, 5]
        self.wagi = [0.05, 0.2, 0.2, 0.1, 0.05, 0.02, 0.08, 0.1, 0.03, 0.17]
    
    def generate_data(self):
        data = {
            "Imię": [random.choices(self.imiona, weights=self.wagi, k=1) for _ in range(self.num_samples)],
            "Nazwisko": [random.choices(self.nazwiska, weights=self.wagi) for _ in range(self.num_samples)],
            "id":[f'P{str(i).zfill(5)}' for i in range(1,self.num_samples + 1)],
            "Miasto": [random.choices(self.miasta, weights=self.wagi) for _ in range(self.num_samples)],
            "ocena_1":[random.choices(self.ocena, weights=self.wagi[::2]) for _ in range(self.num_samples)],
            "ocena_2":[random.choices(self.ocena, weights=self.wagi[2:-3]) for _ in range(self.num_samples)],
            "ocena_3":[random.choices(self.ocena, weights=self.wagi[:-5]) for _ in range(self.num_samples)],
            "ocena_4":[random.choices(self.ocena, weights=self.wagi[5:]) for _ in range(self.num_samples)],
            "ocena_5":[random.choices(self.ocena, weights=self.wagi[::2]) for _ in range(self.num_samples)],
            "Ocena_Koncowa" : [random.choices(self.ocena, self.wagi[:5]) for _ in range(self.num_samples)],
            "Wynik_Egzaminu": [random.randint(50, 100) for _ in range(self.num_samples)]
        }
        return pd.DataFrame(data)


class DataHandler:
    def __init__(self, filename="wyniki.xlsx"):
        self.filename = filename

    def save_to_excel(self, df):
        df.to_excel(self.filename, index=False)
        print(f"Dane zapisane do {self.filename}")

    def read_from_excel(self):
        return pd.read_excel(self.filename)


class DataVisualizer:
    def __init__(self, df):
        self.df = df

    def visualise_Data(self):
        kolumna = df.select_dtypes(include=['object']).columns
        fig = plt.figure(figsize=(25,40))
        for i, cecha in enumerate(kolumna):
            ax = fig.add_subplot(3,4, i+1)
            ax.set_title(cecha)
            (df[cecha].value_counts()/len(df[cecha])).plot.bar()
            ax.set(ylabel="%")
        plt.subplots_adjust


    
    def plot_top_results(self, top_n=10):
        df_sorted = self.df.sort_values(by="Wynik_Egzaminu", ascending=False)
        plt.figure(figsize=(10, 5))
        plt.bar(df_sorted["Nazwisko"].head(top_n), df_sorted["Wynik_Egzaminu"].head(top_n), color='blue')
        plt.xlabel("Nazwisko")
        plt.ylabel("Wynik")
        plt.title(f"Top {top_n} wyników")
        plt.xticks(rotation=45)
        plt.show()
    
    def plot_interpolation(self):
        x = np.arange(0, len(self.df))
        y = self.df["Wynik"].values
        f_interp = interp1d(x, y, kind='linear')
        x_new = np.linspace(0, len(self.df)-1, 200)
        y_new = f_interp(x_new)
        
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, 'o', label="Oryginalne dane")
        plt.plot(x_new, y_new, '-', label="Interpolacja liniowa")
        plt.xlabel("Próbka")
        plt.ylabel("Wynik")
        plt.legend()
        plt.title("Interpolacja wyników")
        plt.show()

class MachineLearning:
    def __init__(self, arr):
        self.arr = arr
    def machoneLearning(self):
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])

        # Model
        model = LinearRegression()
        model.fit(X, y)

        # Predykcja
        print(f"iteracja kolejnych wartości: {X} \n Wartości do szkolenia: {y} \n Model prediction dla wartości 6.: {model.predict([[6]])}")

if __name__ == "__main__":
    generator = DataGenerator()
    df = generator.generate_data()
    
    handler = DataHandler()
    handler.save_to_excel(df)
    
    df_read = handler.read_from_excel()
    
    visualizer = DataVisualizer(df_read)
    visualizer.visualise_Data()
    visualizer.plot_top_results()
    visualizer.plot_interpolation()
    learning = MachineLearning("arr")
    learning.machoneLearning()
