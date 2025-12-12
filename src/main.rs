#![allow(non_snake_case)]
use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, Linear, Module, Optimizer, VarBuilder, VarMap};
const DEVICE: Device = Device::Cpu;

fn main()->anyhow::Result<()>{
    let file = std::fs::File::open("fetch_california_housing.json")?;
    let reader = std::io::BufReader::new(file);

    let data:Data = serde_json::from_reader(reader)?;
    let train_d1 = data.X_train.len();
    let train_d2 = data.X_train[0].len();
    let test_d1 = data.X_test.len();
    let test_d2 = data.X_test[0].len();

    let x_train_vec = data.X_train.into_iter().flatten().collect::<Vec<_>>();
    let x_test_vec = data.X_test.into_iter().flatten().collect::<Vec<_>>();
    let y_train_vec = data.y_train;
    let y_test_vec = data.y_test;

    let x_train = Tensor::from_vec(x_train_vec.clone(), (train_d1,train_d2), &DEVICE)?;
    let y_train = Tensor::from_vec(y_train_vec.clone(),train_d1, &DEVICE)?; 
    let x_test = Tensor::from_vec(x_test_vec.clone(), (test_d1,test_d2), &DEVICE)?;
    let y_test = Tensor::from_vec(y_test_vec.clone(),test_d1, &DEVICE)?; 

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &DEVICE);
    let model = SimpleNN::new(train_d2, vb)?;

    let optim_config = candle_nn::ParamsAdamW{
        lr: 1e-2,
        ..Default::default()
    };
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(),optim_config)?;

    train_model(&model, &x_train, &y_train, &mut optimizer, 200)?;
    evaluate_model(&model, &x_test, &y_test)?;
    Ok(())
}

#[derive(Debug)]
struct SimpleNN{
    fc1: Linear,
    fc2: Linear,
}
impl SimpleNN{
    fn new(in_dim:usize, vb:VarBuilder)->candle_core::Result<Self>{
        let fc1 = linear(in_dim, 64, vb.pp("fc1"))?;
        let fc2 = linear(64, 1, vb.pp("fc2"))?;

        Ok(
            Self{
                fc1,fc2
            }
        )
    }
}
impl Module for SimpleNN{
    fn forward(&self, xs:&Tensor)-> candle_core::Result<Tensor>{
        let x = self.fc1.forward(xs)?;
        let x = x.relu()?;
        let x = self.fc2.forward(&x)?;

        Ok(x)
    }
}

fn train_model(
    model:&SimpleNN, 
    x_train:&Tensor, 
    y_train:&Tensor,
    optimizer:&mut candle_nn::AdamW, 
    epochs:usize
)->anyhow::Result<()>
{
    for epoch in 0..epochs{
        let output = model.forward(x_train)?;
        let loss = candle_nn::loss::mse(&output.squeeze(1)?,y_train)?;
        optimizer.backward_step(&loss)?;
        if(epoch+1)%10==0{
            println!("Epoch {} Train Loss: {}", epoch+1,loss.to_scalar::<f32>()?);
        }
    }
    Ok(())
}

fn evaluate_model(
    model:&SimpleNN,
    x_test:&Tensor,
    y_test:&Tensor,
)->anyhow::Result<()>
{
    let output = model.forward(x_test)?;
    let loss = candle_nn::loss::mse(&output.squeeze(1)?,y_test)?;

    println!("Test loss: {}", loss.to_scalar::<f32>()?);
    Ok(())
}

#[derive(Debug, serde::Deserialize)]
struct Data{
    X_train: Vec<Vec<f32>>,
    X_test: Vec<Vec<f32>>,
    y_train:Vec<f32>,
    y_test:Vec<f32>,
}