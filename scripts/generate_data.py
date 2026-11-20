from loss_landscapes.paths import PROJ_ROOT
from loss_landscapes.datagen.fashion import generate_fashionmnist_data
from loss_landscapes.datagen.func2d import generate_artificial_data
from loss_landscapes.datagen.io import write_data


def main():
    data, targets = generate_fashionmnist_data()
    write_data("ds_fashion.h5", data, targets)
    
    data, targets = generate_artificial_data(n_datapoints = 100_000)
    write_data("ds_func2d.h5", data, targets)
      

if __name__ == "__main__":
    main()
    
