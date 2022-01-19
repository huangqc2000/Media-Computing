python train_img_fitting.py --img_path=./data/knot.jpg --channels=3 --sidelength=512 --compute_grad=False

# camera sine relu tanh sigmoid
python train_img_fitting.py --img_path=./data/camera.jpg --channels=1 --sidelength=256 --compute_grad
python train_img_fitting.py --img_path=./data/camera.jpg --channels=1 --sidelength=256 --compute_grad --nonlinearity=relu --learning_rate=1e-3
python train_img_fitting.py --img_path=./data/camera.jpg --channels=1 --sidelength=256 --compute_grad --nonlinearity=tanh --learning_rate=1e-3
python train_img_fitting.py --img_path=./data/camera.jpg --channels=1 --sidelength=256 --compute_grad --nonlinearity=sigmoid --learning_rate=1e-3


# poisson_equation grad
python train_poisson_grads.py --img_path=./data/starfish.jpg --channels=1 --sidelength=256 --compute_grad
python train_poisson_grads.py --img_path=./data/camera.jpg --channels=1 --sidelength=256

# poisson_equation laplace
python train_poisson_lapl.py --img_path=./data/starfish.jpg --channels=1 --sidelength=256 --compute_grad
python train_poisson_lapl.py --img_path=./data/camera.jpg --channels=1 --sidelength=256

# poisson comp
python train_poisson_compgrad.py --img_path1=./data/gizeh.jpg --img_path2=./data/bear.jpg --channels=1 --sidelength=512

# image inpainting
python train_img_inpainting.py --img_path=./data/knot.jpg --channels=3 --sidelength=512 --num_epochs=100