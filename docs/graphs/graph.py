import matplotlib.pyplot as plt
import argparse

def read_graph(input_file):
    with open(input_file, "r") as file:
        title = file.readline()
        axes = file.readline().split(',')
        xs = [int(x) for x in file.readline().split(",")]
        labels = []
        colors = []
        styles = []
        ys = []
        line = file.readline()
        while line:
            if len(line) == 0:
                line = file.readline()
                continue
            line = line.split(",")
            label = line[0]
            color = line[1]
            style = line[2]
            y = [float(num) for num in line[3:]]
            labels.append(label)
            colors.append(color)
            styles.append(style)
            ys.append(y)
            line = file.readline()

        return title, axes, xs, labels, colors, styles, ys

def plot(title, axes, xs, labels, colors, styles, ys, save_path):
    plt.title(title)
    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    for i in range(len(ys)):
        label = labels[i]
        color = colors[i]
        style = styles[i]
        vals = ys[i]
        plt.plot(xs, vals, label=label, linestyle=style, color=color)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.75)
    plt.savefig(save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    args = parser.parse_args()

    read = read_graph(args.input_file)
    plot(*read, args.output_file)

if __name__ == '__main__':
    main()
    
