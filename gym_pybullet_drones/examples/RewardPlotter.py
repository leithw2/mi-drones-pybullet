import matplotlib.pyplot as plt

class RewardPlotter:
    def __init__(self, title="Progreso del Dron", xlabel="Paso / Episodio", ylabel="Reward"):
        plt.ion()  # Activar modo interactivo
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'r-') # 'r-' es una línea roja
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.rewards = []
        self.steps = []

    def update(self, reward):
        self.rewards.append(reward)
        self.steps.append(len(self.rewards))
        
        # Actualizar datos de la línea
        self.line.set_xdata(self.steps)
        self.line.set_ydata(self.rewards)
        
        # Ajustar límites de los ejes automáticamente
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Dibujar y procesar eventos para que la ventana no se congele
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()