# camera sine relu tanh sigmoid
python train_img_fitting.py --img_path=./data/camera.jpg --channels=1 --sidelength=256 --compute_grad
python train_img_fitting.py --img_path=./data/camera.jpg --channels=1 --sidelength=256 --compute_grad --nonlinearity=relu
python train_img_fitting.py --img_path=./data/camera.jpg --channels=1 --sidelength=256 --compute_grad --nonlinearity=tanh
python train_img_fitting.py --img_path=./data/camera.jpg --channels=1 --sidelength=256 --compute_grad --nonlinearity=sigmoid


# poisson_equation grad
python train_poisson_grads.py --img_path=./data/starfish.jpg --channels=1 --sidelength=256 --compute_grad
python train_poisson_grads.py --img_path=./data/camera.jpg --channels=1 --sidelength=256

# poisson_equation laplace
python train_poisson_lapl.py --img_path=./data/starfish.jpg --channels=1 --sidelength=256 --compute_grad
python train_poisson_lapl.py --img_path=./data/camera.jpg --channels=1 --sidelength=256

# poisson comp
python train_poisson_compgrad.py --img_path1=./data/gizeh.jpg --img_path2=./data/bear.jpg --channels=1 --sidelength=512 --num_epochs=3000
python train_poisson_compgrad.py --img_path1=./data/gizeh.jpg --img_path2=./data/bear.jpg --channels=1 --sidelength=512 --num_epochs=10000
# image inpainting
python train_img_inpainting.py --img_path=./data/butterfly.jpg --channels=3 --sidelength=256
python train_img_inpainting.py --img_path=./data/butterfly.jpg --channels=3 --sidelength=256 --points_choose=1000
python train_img_inpainting.py --img_path=./data/butterfly.jpg --channels=3 --sidelength=256 --points_choose=5000
python train_img_inpainting.py --img_path=./data/butterfly.jpg --channels=3 --sidelength=256 --points_choose=10000
python train_img_inpainting.py --img_path=./data/butterfly.jpg --channels=3 --sidelength=256 --points_choose=30000

python train_img_inpainting.py --img_path=./data/human.jpg --channels=3 --sidelength=256
python train_img_inpainting.py --img_path=./data/human.jpg --channels=3 --sidelength=256 --points_choose=1000
python train_img_inpainting.py --img_path=./data/human.jpg --channels=3 --sidelength=256 --points_choose=5000
python train_img_inpainting.py --img_path=./data/human.jpg --channels=3 --sidelength=256 --points_choose=10000
python train_img_inpainting.py --img_path=./data/human.jpg --channels=3 --sidelength=256 --points_choose=30000