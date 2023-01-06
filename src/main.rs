use std::fs;
use serde_json::json;
use serde_json::{ Value };

use peroxide::prelude::*;
use peroxide::structure::matrix::Matrix;
use peroxide::numerical::eigen::{ eigen, EigenMethod };

use pyo3::prelude::*;
use peroxide::util::plot::*;

fn parse_json_sources() -> [ Vec<[f64; 2]> ; 4 ]{
    let source_eth_usdc = fs::read_to_string("sources/univ2-eth-usdc.json");
    let source_eth_torn = fs::read_to_string("sources/univ2-eth-torn.json");

    let data_eth_usdc: Value = serde_json::from_str(
        &source_eth_usdc.unwrap()
    ).unwrap();
    let data_eth_torn: Value = serde_json::from_str(
        &source_eth_torn.unwrap()
    ).unwrap();

    let historic_eth_usdc = data_eth_usdc["data"]["pairDayDatas"]
        .as_array().unwrap();
    let historic_eth_torn = data_eth_torn["data"]["pairDayDatas"]
        .as_array().unwrap();
    let set_size = usize::clone(&historic_eth_torn.len());

    let reserves_torn_eth: Vec<[f64; 2]> = historic_eth_torn.iter()
        .map(|e| [
            e["reserve0"].as_str().unwrap().parse::<f64>().unwrap(),
            e["reserve1"].as_str().unwrap().parse::<f64>().unwrap()
        ]).collect();
    let volumes_torn_eth: Vec<[f64; 2]> = historic_eth_torn.iter()
        .map(|e| [
            e["dailyVolumeToken0"].as_str().unwrap().parse::<f64>().unwrap(),
            e["dailyVolumeToken1"].as_str().unwrap().parse::<f64>().unwrap()
        ]).collect();
    let price_torn_eth: Vec<[f64; 2]> = historic_eth_torn.iter()
        .map(|e| [
            e["date"].as_f64().unwrap(),
            ( e["reserve1"].as_str().unwrap().parse::<f64>().unwrap() /
              e["reserve0"].as_str().unwrap().parse::<f64>().unwrap() )
         ]).collect();
    let price_eth_usdc: Vec<[f64; 2]> = historic_eth_usdc.iter()
        .map(|e| [
            e["date"].as_f64().unwrap(),
            ( e["reserve0"].as_str().unwrap().parse::<f64>().unwrap() /
              e["reserve1"].as_str().unwrap().parse::<f64>().unwrap() )
        ]).collect();

    [
        volumes_torn_eth,
        reserves_torn_eth,
        price_torn_eth,
        price_eth_usdc[0..set_size].to_vec()
    ]
}

fn timeseries_eth_denom_to_usd(
    eth_historic: Vec<[f64; 2]>,
    timeseries: Vec<[f64; 2]>
) -> Vec<[f64; 2]> {
    let mut timeseries_usd: Vec<[f64; 2]> = Vec::new();

    for x in 0..timeseries.len()
    {
        timeseries_usd.push([
            timeseries[x][0],
            eth_historic[x][1] * timeseries[x][1]
        ]);
    }

    timeseries_usd
}

fn series_volume_usd(
    eth_historic_usd: Vec<[f64; 2]>,
    token_historic_usd: Vec<[f64; 2]>,
    volume_historic: Vec<[f64; 2]>
) -> Vec<[f64; 2]> {
    let mut series_volume_usd: Vec<[f64; 2]> = Vec::new();

    for x in 0..volume_historic.len()
    {
        series_volume_usd.push([
            eth_historic_usd[x][1] * volume_historic[x][0],
            token_historic_usd[x][1] * volume_historic[x][1]
        ]);
    }

    series_volume_usd
}

fn covariance_matrix(
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    a_mean: f64,
    b_mean: f64,
    c_mean: f64
) -> Matrix {
    let mut variance_matrix: Matrix = zeros(3, 3);
    let a_variance = variance(a.clone(), f64::clone(&a_mean));
    let b_variance = variance(b.clone(), f64::clone(&b_mean));
    let c_variance = variance(c.clone(), f64::clone(&c_mean));

    let covariance_ab = covariance(
        a.clone(), b.clone(), f64::clone(&a_mean), f64::clone(&b_mean)
    );
    let covariance_ac = covariance(
        a.clone(), c.clone(), f64::clone(&a_mean), f64::clone(&c_mean)
    );
    let covariance_bc = covariance(
        b.clone(), c.clone(), f64::clone(&b_mean), f64::clone(&c_mean)
    );

    variance_matrix[(0, 0)] = a_variance;
    variance_matrix[(1, 1)] = b_variance;
    variance_matrix[(2, 2)] = c_variance;
    variance_matrix[(0, 1)] = covariance_ab;
    variance_matrix[(0, 2)] = covariance_ac;
    variance_matrix[(1, 0)] = covariance_ab;
    variance_matrix[(1, 2)] = f64::clone(&covariance_bc);
    variance_matrix[(2, 0)] = f64::clone(&covariance_ac);
    variance_matrix[(2, 1)] = f64::clone(&covariance_bc);

    variance_matrix
}

fn variance(
    a: Vec<f64>,
    a_mean: f64
) -> f64 {
    let mut variance: f64 = 0.;

    for x in 0..a.len()
    {
        variance = (a[x] - a_mean).powi(2);
    }

    variance = variance / ((a.len() - 1) as f64);
    variance
}

fn covariance(
    a: Vec<f64>,
    b: Vec<f64>,
    a_mean: f64,
    b_mean: f64
) -> f64 {
    let mut covariance: f64 = 0.;

    for x in 0..b.len()
    {
        covariance = (a[x] - a_mean) * (b[x] - b_mean);
    }

    covariance = covariance / ((b.len() - 1) as f64);
    covariance
}

fn beta_criterion(
    allocation: f64,
    covariance: Matrix,
    mean_a: f64,
    mean_b:f64,
    mean_c: f64
) -> [f64; 3] {
    let [ var_a, var_b, var_c ] = [
        covariance[(0, 0)], covariance[(1, 1)], covariance[(2, 2)] 
    ];
    let [ cov_ab, cov_bc, cov_ca ] = [
        covariance[(0, 1)], covariance[(1, 2)], covariance[(0, 2)]   
    ];
 
    let beta_ab = ((cov_ab / var_a) + (cov_ab / var_b )).abs();
    let beta_bc = ((cov_bc / var_b) + (cov_bc / var_c )).abs();
    let beta_ca = ((cov_ca / var_a) + (cov_ca/ var_b )).abs();

    let beta_sum = beta_ab + beta_bc + beta_ca;

    let a_allocation = allocation * ((beta_ab) / beta_sum);
    let b_allocation = allocation * ((beta_bc) / beta_sum);
    let z_allocation = allocation * ((beta_ca) / beta_sum);

    [ a_allocation, b_allocation, z_allocation ]
}

fn criterion(
    total_allocation: f64,
    ev_set: [f64 ; 3],
    exclusion_index: usize
) -> [f64 ; 3] {
    let mut ev_select = ev_set.clone();

    let sum_of_ratios: f64 = ev_select.iter().sum();
    let allocation_a = total_allocation * (ev_select[0] / sum_of_ratios);
    let allocation_b = total_allocation * (ev_select[1] / sum_of_ratios);
    let allocation_c = total_allocation * (ev_select[2] / sum_of_ratios);

   let mut allocations = [
        allocation_a,
        allocation_b,
        allocation_c
    ];

    allocations
}

fn mean(a: Vec<f64>) -> f64 {
    let mut mean = 0.;

    for x in 0..a.len() { mean += a[x]; }

    mean = mean / (a.len() as f64);
    mean
}

fn volatility_and_snr(
    a_variance: f64,
    a_mean: f64,
    period: f64
) -> [f64; 2] {
    let std_dev = a_variance.sqrt();
    let volatility_index = (std_dev * period.sqrt()) * 100.0;
    let cv = a_variance.sqrt() / a_mean;
    let snr = volatility_index / cv;

    [ volatility_index, snr ]
}

fn ev_index(
    covariance: &Matrix,
    eigenvector: &Matrix,
) -> [f64 ; 3] {

    let mut eigen_weights: Matrix = zeros(3, 1);

    eigen_weights[(0, 0)] = 1.00;
    eigen_weights[(1, 0)] = 1.00;
    eigen_weights[(2, 0)] = 1.00;

    let eigen_portfolio = eigenvector * &eigen_weights;
    let ev_sum = eigen_portfolio[(0, 0)]
        + eigen_portfolio[(1, 0)] + 
        eigen_portfolio[(2, 0)];

    let ev_a = eigen_portfolio[(0,0)] / ev_sum;
    let ev_b = eigen_portfolio[(1,0)] / ev_sum;
    let ev_c = eigen_portfolio[(2,0)] / ev_sum;

    [ ev_a, ev_b, ev_c ]
}

fn log_returns(
    index: usize,
    source_set: Vec<f64> 
) -> f64 { 
    let mut log_returns = 0.;

    if index != 0 
    {
        let previous_returns = source_set[index - 1];

        if previous_returns != 0.0 
        {
            log_returns = (source_set[index] / previous_returns).ln();
        }
    } 

    log_returns
}

fn simulate_lp_pos( 
    usd_series: Vec<f64>,
    vol_series: Vec<[f64; 2]>,
    index: usize,
) -> [f64 ; 2] {
    let mut prev_pos_price = 0.0;
    let pos_price = usd_series[index];

    if index != 0
    {
        prev_pos_price = usd_series[index - 1];
    }

    let delta_u = pos_price - prev_pos_price;
    let delta_d = prev_pos_price - pos_price;

    let mut delta_pos = 0.0;
    let mut fees_pos = [ 0.0, 0.0 ];

    if delta_u > 0.0 
    {
       delta_pos = delta_u / prev_pos_price; 
    } 
    else if delta_d > 0.0 
    { 
       delta_pos = delta_d / prev_pos_price;
    } 

    let fee_coffeicient = 0.003;
    let assumption_coefficient = 0.25;

    if delta_pos > 0.0199
    {
        let vol_a = vol_series[index][0] * assumption_coefficient;
        let vol_b = vol_series[index][1] * assumption_coefficient;

        fees_pos = [ 
            vol_a * fee_coffeicient,
            vol_b * fee_coffeicient
        ];
    } 
    fees_pos
}

fn benchmark_strategy(strategy_allocation: f64) {
    let [ volumes, liquidity, torn_eth, eth_usd_series ] = parse_json_sources();

    let eth_usd: Vec<f64> = eth_usd_series.iter().map(|e| e[1]).collect();
    let volumes_usd = series_volume_usd(
        eth_usd_series.clone(), 
        Vec::clone(&torn_eth), 
        volumes.clone()
    );
    let torn_usd: Vec<f64> = timeseries_eth_denom_to_usd(
        eth_usd_series.clone(), 
        torn_eth
    ).iter().map(|e| e[1]).collect();

    let log_eth_usd: Vec<f64> = eth_usd.iter().enumerate()
        .map(|(i, e)| log_returns(i, eth_usd.clone()))
        .collect();
    let log_torn_usd: Vec<f64> = torn_usd.iter().enumerate()
        .map(|(i, e)| log_returns(i, torn_usd.clone()))
        .collect();

    let series_set_size = usize::clone(&torn_usd.len());

    let lp_d_pos: Vec<[f64; 2]> = eth_usd.clone().iter().enumerate()
        .map(|(i, _)| simulate_lp_pos(eth_usd.clone(), volumes.clone(), i ))
        .collect();
    let lp_u_pos: Vec<[f64 ; 2]> = eth_usd.clone().iter().enumerate()
        .map(|(i, _)| simulate_lp_pos(torn_usd.clone(), volumes.clone(), i))
        .collect();    
    let lp_pos: Vec<[f64 ; 2]> = volumes.clone().iter().enumerate()
        .map(|(i, e)| [ e[0] * 0.003, e[1] * 0.003 ])
        .collect(); 

    let llp_pos: Vec<f64> = lp_pos.clone().iter().enumerate()
        .map(|(i, e)| (
            eth_usd[series_set_size - 1] * e[1] + 
            torn_usd[series_set_size - 1] * e[0] 
        )).collect();

    let total_lp_pos: Vec<f64> = lp_pos.clone().iter().enumerate()
        .map(|(i, e)| (
            eth_usd[series_set_size - 1] * lp_u_pos[i][1] +
            torn_usd[series_set_size - 1] * lp_u_pos[i][0] +
            eth_usd[series_set_size - 1] * lp_d_pos[i][1] +
            torn_usd[series_set_size - 1] * lp_d_pos[i][0] +
            eth_usd[series_set_size - 1] * e[1] + 
            torn_usd[series_set_size - 1] * e[0] 
        )).collect();

    let lp_pos_usd: Vec<f64> = total_lp_pos.clone().iter().enumerate()
        .into_iter()
        .scan(0.0, |acc, (i, e)| {
            *acc += e;
            Some(*acc)
        }).collect();

    let llp_pos_usd: Vec<f64> = llp_pos.clone().iter().enumerate()
        .into_iter()
        .scan(0.0, |acc, (i, e)| {
            *acc += e;
            Some(*acc)
        }).collect();

    let log_lp_usd: Vec<f64> = lp_pos_usd.iter().enumerate()
        .map(|(i, e)| log_returns(i, lp_pos_usd.clone()))
        .collect();

    let mut series_ev: Vec<[f64; 3]> = vec![];
    let mut series_cov: Vec<[f64; 3]>= vec![];
    let mut series_evv: Vec<[f64; 3]>= vec![];

    for n in 2..series_set_size
    {    
        let slice_log_torn_usd = log_torn_usd[0..n].to_vec();
        let slice_log_eth_usd = log_eth_usd[0..n].to_vec();
        let slice_log_lp_usd = log_lp_usd[0..n].to_vec();

        let log_torn_usd_mean = mean(slice_log_torn_usd.clone());
        let log_eth_usd_mean = mean(slice_log_eth_usd.clone());
        let log_lp_usd_mean = mean(slice_log_lp_usd.clone());

        let covariance_m3 = covariance_matrix(
            slice_log_eth_usd,
            slice_log_torn_usd,
            slice_log_lp_usd,
            log_eth_usd_mean,
            log_torn_usd_mean,
            log_lp_usd_mean
        );

        let lmda = eigen(&covariance_m3, EigenMethod::Jacobi);
        let ev_set = ev_index(&covariance_m3, &lmda.eigenvector);

        series_cov.push([ 
            covariance_m3[(0, 1)], 
            covariance_m3[(1, 2)], 
            covariance_m3[(0, 2)]  
        ]);
        series_ev.push([
            lmda.eigenvalue[0],
            lmda.eigenvalue[1],
            lmda.eigenvalue[2]
        ]);
        series_evv.push(ev_set);
   }

    plot_figure(
        1,
        &eth_usd_series,
        &series_ev,
        vec![ "TIME (MS)", "EIGENVALUE (EV)" ],
        vec![ "", "TORN", "UNIV2-2FOLD" ],
        "Line",
        "./ev-eth-torn-univ2_fold.png",
        14,
        3
    );

    plot_figure(
        0,
        &eth_usd_series,
        &series_cov,
        vec![ "TIME (MS)", "COVARIANCE (CV)", "" ],
        vec![ "(ETH, TORN)", "(TORN, UNIV2-2FOLD)", "(UNIV2-2FOLD, ETH)" ],
        "Point",
        "./cov-eth-torn-univ2_fold.png",
        18,
        3
    );

    let lp_series: Vec<[f64; 3]> = lp_pos_usd.clone().iter().enumerate()
        .map(|(i, e)| [ 
            f64::clone(&volumes_usd[i][1]) * (-1.00),
            f64::clone(&llp_pos_usd[i]), 
            f64::clone(&lp_pos_usd[i]), 
        ])
        .collect();

    plot_figure(
        1,
        &eth_usd_series,
        &lp_series,
        vec![ "TIME (MS)", "VALUE (USD)" ],
        vec![ "", "UNIV2-ETH-TORN", "UNIV2-2FOLD" ],
        "Line",
        "./culm-fees_univ2-eth-torn_2fold.png",
        0,
        3
    );

    let evv_series: Vec<[f64; 3]> = series_evv.clone().iter().enumerate()
        .map(|(i, e)| [ 
            e[0] + 1.00,
            e[1] - 1.00,  
            e[2] 
        ])
        .collect();

    plot_figure(
        0,
        &eth_usd_series,
        &evv_series,
        vec![ "TIME(MS)", "EIGENVECTOR (EVV)" ],
        vec![ "ETH", "TORN", "UNIV2-2FOLD" ],
        "Line",
        "./evv-eth-torn-univ2_2fold.png",
        100,
        3
    );

    let evv_price_series: Vec<[f64; 3]> = series_evv.clone().iter().enumerate()
        .map(|(i, e)| [ 
            eth_usd[i] / 10.0,
            torn_usd[i],  
            series_evv[i][2 ] * -150.00 
        ])
    .collect();

    plot_figure(
        0,
        &eth_usd_series,
        &evv_price_series,
        vec![ "TIME(MS)", "VALUE (USD)" ],
        vec![ "ETH", "TORN", "UNIV2-2FOLD EV" ],
        "Line",
        "./eth-torn-usd-univ2_2fold-evv.png",
        0,
        3
    );

    let fees_volume_series: Vec<[f64; 3]> = series_evv.clone().iter().enumerate()
        .map(|(i, e)| [ 
            series_evv[i][2] * - 100000000.0, 
            series_evv[i][2] * - 100000000.0, 
            volumes_usd[i + 2][1] + volumes_usd[i + 2][0],  
        ])
    .collect();

    plot_figure(
        1,
        &eth_usd_series,
        &fees_volume_series,
        vec![ "TIME(MS)", "VALUE (USD)" ],
        vec![ "UNIV2-2FOLD EVV", "UNIV2-2FOLD EVV", "VOLUME" ],
        "Line",
        "./eth-torn-usd-univ2_2fold-vol-fees.png",
        0,
        3
    );

    let eth_mean = mean(log_eth_usd.clone());
    let torn_mean = mean(log_torn_usd.clone());
    let lp_mean = mean(log_lp_usd.clone());

    let eth_variance = variance(log_eth_usd.clone(), eth_mean);
    let torn_variance = variance(log_torn_usd.clone(), torn_mean);
    let lp_variance = variance(log_lp_usd.clone(), lp_mean);

    let [ eth_volatility_index, eth_snr ] = volatility_and_snr(
        f64::clone(&eth_variance),
        f64::clone(&eth_mean),
        f64::clone(&(series_set_size as f64))
    );
    let [ torn_volatility_index, torn_snr ] = volatility_and_snr(
        f64::clone(&torn_variance),
        f64::clone(&torn_mean),
        f64::clone(&(series_set_size as f64))
    );
    let [ lp_volatility_index, lp_snr ] = volatility_and_snr(
        f64::clone(&lp_variance),
        f64::clone(&lp_mean),
        f64::clone(&(series_set_size as f64))
    );

    println!("ETH");
    println!("  λ: {:?}", series_ev[series_ev.len() - 1][0]);
    println!("  VI: {:?}", eth_volatility_index);
    println!("  S/N: {:?}", eth_snr);
    println!("  EVV: {:?}", series_evv[series_evv.len() - 1][0]);

    println!("TORN");
    println!("  λ: {:?}", series_ev[series_ev.len() - 1][1]);
    println!("  VI: {:?}", torn_volatility_index);
    println!("  S/N: {:?}", torn_snr);
    println!("  EVV: {:?}", series_evv[series_evv.len() - 1][1]);

    println!("UNIV2-2FOLD");
    println!("  λ: {:?}", series_ev[series_ev.len() - 1][2]);
    println!("  VI: {:?}", lp_volatility_index);
    println!("  S/N: {:?}", lp_snr);
    println!("  EVV: {:?}", series_evv[series_evv.len() - 1][2]);

    let [ ev_alloc_a, ev_alloc_b, ev_alloc_c ] = criterion(
        strategy_allocation,
        series_evv[series_evv.len() - 1],
        0
    );

    println!("ALLOCATION");
    println!("EVV");    
    println!("  ${:?} in ETH", ev_alloc_a);
    println!("  ${:?} in TORN", ev_alloc_b);
    println!("  ${:?} in UNIV2-ETH-TORN", ev_alloc_c);

} 

fn plot_figure(
    domain_index: usize,
    x_domain: &Vec<[f64 ; 2]>,
    y_domains: &Vec<[f64; 3]>,
    labels: Vec<&str>,
    legend: Vec<&str>,
    fig_type: &str,
    filepath: &str, 
    offset: usize,
    domain_size: usize
) { 
   let mut figure = Plot2D::new(); 
   let mut figure_plot_type = vec![ Line, Line, Line ];

    if fig_type == "Point" {
     figure_plot_type = vec![ Point, Point, Point ];
   } 

    figure.set_domain(
        x_domain[offset..y_domains.len()].iter()
        .map(|e| e[0]).collect()
    );

    for i in domain_index..domain_size 
    {
        figure.insert_image(
            y_domains[offset..y_domains.len()].iter()
            .map(|e| e[i]).collect()
        );
    }

    figure
    .set_title("")
    .set_fig_size((10, 6))
    .set_marker(
        Vec::clone(
            &[ &figure_plot_type[domain_index..domain_size] ]
            .concat().to_vec()
        )
    )
    .set_legend(
        Vec::clone(
            &[ &legend[domain_index..domain_size] ]
            .concat().to_vec()
        )
    )
    .set_dpi(300)
    .set_path(filepath)
    .set_ylabel(labels[1])
    .set_xlabel(labels[0])
    .savefig();
}

#[test]
fn main() {
    pyo3::prepare_freethreaded_python();
 
    benchmark_strategy(500000.00)
}