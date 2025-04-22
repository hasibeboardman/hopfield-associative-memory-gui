import tkinter as tk
import numpy as np

class HopfieldNN:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, pattern):
        pattern = np.array(pattern)
        pattern[pattern == 0] = -1
        self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern):
        pattern = np.array(pattern)
        pattern[pattern == 0] = -1
        result = np.sign(np.dot(self.weights, pattern))
        result[result == -1] = 0
        return result.tolist()

class HopfieldGUI:
    def __init__(self, master, size=10):
        self.master = master
        self.size = size
        self.pattern_size = size * size
        self.network = HopfieldNN(self.pattern_size)

        self.status_label = tk.Label(master, text="Status: Ready")
        self.status_label.pack()

        learned_patterns_frame = tk.Frame(master)
        learned_patterns_frame.pack(side=tk.LEFT, padx=10)

        query_frame = tk.Frame(master)
        query_frame.pack(side=tk.LEFT, padx=20)

        result_frame = tk.Frame(master)
        result_frame.pack(side=tk.LEFT, padx=20)

        # Create three canvases for learned patterns
        self.learned_canvases = []
        for i in range(3):
            canvas = self.create_canvas(learned_patterns_frame, f"Learned Pattern {i + 1}", canvas_size=200)
            self.learned_canvases.append(canvas)

        # Create one canvas for the query pattern
        self.query_canvas = self.create_canvas(query_frame, "Query Pattern")

        # Create one canvas for the result pattern
        self.result_canvas = self.create_canvas(result_frame, "Result Pattern", enable_draw=False, canvas_size=200)

        learn_button = tk.Button(learned_patterns_frame, text="Learn", command=self.learn)
        learn_button.pack()

        query_button = tk.Button(query_frame, text="Query", command=self.query)
        query_button.pack()

        clear_button = tk.Button(master, text="Clear", command=self.clear)
        clear_button.pack()

        show_pattern_button = tk.Button(result_frame, text="Show Learned Pattern", command=self.show_learned_pattern)
        show_pattern_button.pack()

        self.result_canvas.bind("<Button-1>", lambda event: self.show_learned_pattern())

    def create_canvas(self, frame, title, enable_draw=True, canvas_size=400):
        label = tk.Label(frame, text=title)
        label.pack()

        canvas = tk.Canvas(frame, width=canvas_size, height=canvas_size, bg='white')
        canvas.pack()

        for i in range(self.size):
            for j in range(self.size):
                x0, y0 = j * (canvas_size // self.size), i * (canvas_size // self.size)
                x1, y1 = x0 + (canvas_size // self.size), y0 + (canvas_size // self.size)
                canvas.create_rectangle(x0, y0, x1, y1, fill="white", tags="rect")

        if enable_draw:
            canvas.bind("<B1-Motion>", lambda event, c=canvas: self.draw(event, c))
            canvas.bind("<Button-1>", lambda event, c=canvas: self.draw(event, c))

        return canvas

    def draw(self, event, canvas):
        x, y = event.x, event.y
        items = canvas.find_closest(x, y)
        current_color = canvas.itemcget(items[0], "fill")
        new_color = "black" if current_color == "white" else "white"
        canvas.itemconfig(items[0], fill=new_color)

    def learn(self):
        for i, canvas in enumerate(self.learned_canvases):
            pattern = self.get_pattern(canvas)
            if pattern.count(1) > 0:
                self.network.train(pattern)
                self.status_label.config(text=f"Status: Pattern {i + 1} learned")
            else:
                self.status_label.config(text=f"Status: Nothing to learn in Pattern {i + 1}")

    def query(self):
        pattern = self.get_pattern(self.query_canvas)
        result_pattern = self.network.recall(pattern)
        self.display_pattern(self.result_canvas, result_pattern, query_pattern=pattern)
        self.status_label.config(text="Status: Query result displayed")

    def get_pattern(self, canvas):
        pattern = []
        for item in canvas.find_all():
            color = canvas.itemcget(item, "fill")
            pattern.append(1 if color == "black" else 0)
        return pattern

    def display_pattern(self, canvas, pattern, query_pattern=None):
        canvas.delete("all")
        matched_learned_pattern = None
        for i, learned_canvas in enumerate(self.learned_canvases):
            learned_pattern = np.array(self.get_pattern(learned_canvas))
            query_pattern_array = np.array(query_pattern) if query_pattern else None

            hamming_distance = np.sum(query_pattern_array != learned_pattern) if query_pattern_array is not None else 0

            if hamming_distance <= int(0.1 * len(learned_pattern)):
                matched_learned_pattern = learned_pattern
                break

        if matched_learned_pattern is not None:
            for i in range(self.size):
                for j in range(self.size):
                    idx = i * self.size + j
                    color = "black" if matched_learned_pattern[idx] == 1 else "white"
                    x0, y0 = j * (canvas.winfo_width() // self.size), i * (canvas.winfo_height() // self.size)
                    x1, y1 = x0 + (canvas.winfo_width() // self.size), y0 + (canvas.winfo_height() // self.size)
                    canvas.create_rectangle(x0, y0, x1, y1, fill=color, tags="rect")
        else:
            self.status_label.config(text="Status: No matching learned pattern found")

    def clear(self):
        for canvas in [self.query_canvas, self.result_canvas] + self.learned_canvases:
            for item in canvas.find_all():
                canvas.itemconfig(item, fill="white")
        self.status_label.config(text="Status: Cleared")

    def show_learned_pattern(self):
        query_pattern = np.array(self.get_pattern(self.query_canvas))

        for i, canvas in enumerate(self.learned_canvases):
            learned_pattern = np.array(self.get_pattern(canvas))
            hamming_distance = np.sum(query_pattern != learned_pattern)

            if hamming_distance <= int(0.1 * len(learned_pattern)):
                self.display_pattern(self.result_canvas, learned_pattern)
                self.status_label.config(text=f"Status: Learned pattern {i + 1} displayed")
                return

        self.status_label.config(text="Status: No matching learned pattern found")

def main():
    root = tk.Tk()
    root.title("Hopfield Network GUI")
    app = HopfieldGUI(root, size=10)
    root.mainloop()

if __name__ == "__main__":
    main()





