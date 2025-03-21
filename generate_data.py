import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class DataGenerator:
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.imiona = ["Jan", "Anna", "Piotr", "Katarzyna", "Marek", "Maria", "Tomasz", "Agnieszka", "Paweł", "Joanna"]
        self.nazwiska = ["Kowalski", "Nowak", "Wiśniewski", "Wójcik", "Kamiński", "Lewandowski", "Zieliński", "Szymański", "Dąbrowski", "Woźniak"]
        self.miasta = ["Warszawa", "Kraków", "Gdańsk", "Wrocław", "Poznań", "Łódź", "Szczecin", "Lublin", "Katowice", "Bydgoszcz"]
    
    def generate_data(self):
        data = {
            "Imię": [random.choice(self.imiona) for _ in range(self.num_samples)],
            "Nazwisko": [random.choice(self.nazwiska) for _ in range(self.num_samples)],
            "Miasto": [random.choice(self.miasta) for _ in range(self.num_samples)],
            "Wynik": [random.randint(50, 100) for _ in range(self.num_samples)]
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
    
    def plot_top_results(self, top_n=10):
        df_sorted = self.df.sort_values(by="Wynik", ascending=False)
        plt.figure(figsize=(10, 5))
        plt.bar(df_sorted["Nazwisko"].head(top_n), df_sorted["Wynik"].head(top_n), color='blue')
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


if __name__ == "__main__":
    generator = DataGenerator()
    df = generator.generate_data()
    
    handler = DataHandler()
    handler.save_to_excel(df)
    
    df_read = handler.read_from_excel()
    
    visualizer = DataVisualizer(df_read)
    visualizer.plot_top_results()
    visualizer.plot_interpolation()
