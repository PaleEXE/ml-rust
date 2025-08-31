use std::fmt::Display;

struct Norm {
    min: f64,
    max: f64,
}

impl Norm {
    fn new() -> Self {
        Norm { min: 0.0, max: 0.0 }
    }

    fn normalize(&mut self, x: &[f64]) -> Vec<f64> {
        let min = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        self.max = max;
        self.min = min;
        x.iter().map(|&v| (v - min) / (max - min)).collect()
    }
}
struct LinearRegression {
    slope: f64,
    intercept: f64,
    iters: usize,
    learning_rate: f64,
    x_norm: Norm,
    y_norm: Norm,
}

impl LinearRegression {
    fn new(iters: usize, learning_rate: f64) -> Self {
        Self {
            slope: 0.0,
            intercept: 0.0,
            x_norm: Norm::new(),
            y_norm: Norm::new(),
            iters,
            learning_rate,
        }
    }

    fn mse(&self, x: &[f64], y: &[f64]) -> f64 {
        let (s, i) = self.denormalize_params();

        x.iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| {
                let pred = s * xi + i;
                (pred - yi).powi(2)
            })
            .sum::<f64>()
            / x.len() as f64
    }

    fn denormalize_params(&self) -> (f64, f64) {
        let (x_min, x_range) = (self.x_norm.min, self.x_norm.max - self.x_norm.min);
        let (y_min, y_range) = (self.y_norm.min, self.y_norm.max - self.y_norm.min);

        (
            self.slope * (y_range / x_range),
            self.intercept * y_range + y_min - self.slope * x_min,
        )
    }

    fn fit(&mut self, x: &[f64], y: &[f64]) {
        let x_norm = self.x_norm.normalize(x);
        let y_norm = self.y_norm.normalize(y);
        let n = x_norm.len() as f64;
        self.intercept = x_norm.iter().sum::<f64>() / n;

        for _ in 0..self.iters {
            let (mut slope_grad, mut intercept_grad) = (0.0, 0.0);

            for i in 0..x_norm.len() {
                let pred = self.slope * x_norm[i] + self.intercept;
                let error = pred - y_norm[i];
                slope_grad += x_norm[i] * error;
                intercept_grad += error;
            }

            slope_grad *= 2.0 / n;
            intercept_grad *= 2.0 / n;

            self.slope -= self.learning_rate * slope_grad;
            self.intercept -= self.learning_rate * intercept_grad;

            println!("MSE = {:.6}", self.mse(x, y));
        }
    }
}

fn main() {
    let xs: Vec<f64> = (0..100000).map(|x| x as f64).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| 100.0 * x + x.sin()).collect();

    let mut lr = LinearRegression::new(1000, 0.7);
    lr.fit(&xs, &ys);

    let (s, i) = lr.denormalize_params();
    println!("Final slope = {:.6}\n intercept = {:.6}", s, i);
}
