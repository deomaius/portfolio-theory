use std::fs;

use serde_json::{ Error, Value };
use serde_json::json;

use ndarray::{ Array2, Array3 };

fn parse_json_sources() -> [ Vec<[f64; 2]> ; 5 ]{
    let source_eth_dai = fs::read_to_string("sources/univ2-eth-dai.json");
    let source_eth_torn = fs::read_to_string("sources/univ2-eth-torn.json");

    let data_eth_dai: Value = serde_json::from_str(
        &source_eth_dai.unwrap()
    ).unwrap();
    let data_eth_torn: Value = serde_json::from_str(
        &source_eth_torn.unwrap()
    ).unwrap();

    let historic_eth_dai = data_eth_dai["data"]["pairDayDatas"]
        .as_array().unwrap();
    let historic_eth_torn = data_eth_torn["data"]["pairDayDatas"]
        .as_array().unwrap();
    let set_size = usize::clone(&historic_eth_torn.len());

    let reserves_torn_eth: Vec<[f64; 2]> = historic_eth_torn.iter()
        .map(|e| [
            e["reserve1"].as_str().unwrap().parse::<f64>().unwrap(),
            e["reserve0"].as_str().unwrap().parse::<f64>().unwrap()
        ]).collect();
    let volumes_torn_eth: Vec<[f64; 2]> = historic_eth_torn.iter()
        .map(|e| [
            e["dailyVolumeToken1"].as_str().unwrap().parse::<f64>().unwrap(),
            e["dailyVolumeToken0"].as_str().unwrap().parse::<f64>().unwrap()
        ]).collect();
    let supply_lp_token: Vec<[f64; 2]> = historic_eth_torn.iter()
        .map(|e| [
            e["date"].as_f64().unwrap(),
            e["totalSupply"].as_str().unwrap().parse::<f64>().unwrap()
        ]).collect();
    let price_torn_eth: Vec<[f64; 2]> = historic_eth_torn.iter()
        .map(|e| [
            e["date"].as_f64().unwrap(),
            ( e["reserve1"].as_str().unwrap().parse::<f64>().unwrap() /
              e["reserve0"].as_str().unwrap().parse::<f64>().unwrap() )
         ]).collect();
    let price_eth_dai: Vec<[f64; 2]> = historic_eth_dai.iter()
        .map(|e| [
            e["date"].as_f64().unwrap(),
            ( e["reserve0"].as_str().unwrap().parse::<f64>().unwrap() /
              e["reserve1"].as_str().unwrap().parse::<f64>().unwrap() )
        ]).collect();

    [
        supply_lp_token,
        volumes_torn_eth,
        reserves_torn_eth,
        price_torn_eth,
        price_eth_dai[0..set_size].to_vec()
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
) -> Array3<f64> {
    let mut variance_matrix = Array3::<f64>::zeros((3, 3, 1));
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

    variance_matrix[[0, 0, 0]] = a_variance;
    variance_matrix[[1, 1, 0]] = b_variance;
    variance_matrix[[2, 2, 0]] = c_variance;
    variance_matrix[[0, 1, 0]] = covariance_ab;
    variance_matrix[[0, 2, 0]] = covariance_ac;
    variance_matrix[[1, 0, 0]] = covariance_ab;
    variance_matrix[[1, 2, 0]] = f64::clone(&covariance_bc);
    variance_matrix[[2, 0, 0]] = f64::clone(&covariance_ab);
    variance_matrix[[2, 1, 0]] = f64::clone(&covariance_ac);

    println!("VAR(x): {:?}", a_variance);
    println!("VAR(y): {:?}", b_variance);
    println!("VAR(z): {:?}", c_variance);
    println!("COV(x,y): {:?}", covariance_ab);
    println!("COV(x,z): {:?}", covariance_ac);
    println!("COV(y,z): {:?}", covariance_bc);

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
    covariance: Array2<f64>
) -> [f64; 2] {
    let std_a = covariance[[0, 0]].sqrt();
    let std_b = covariance[[1, 1]].sqrt();
    let beta_a = covariance[[0, 1]] * (std_a / std_b);
    let beta_b = covariance[[1, 0]] * (std_b / std_a);

    println!("BETA A: {:?}", beta_a);
    println!("BETA B: {:?}", beta_b);

    let beta_sum = beta_a + beta_b;

    println!("BETA A MULTIPLIER: {:?}", (beta_a / beta_sum));
    println!("BETA B MULTIPLIER: {:?}", (beta_b / beta_sum));

    let a_allocation = allocation * (beta_a / beta_sum);
    let b_allocation = allocation * (beta_b / beta_sum);

    [ a_allocation, b_allocation ]
}

fn snr_criterion(
    allocation: f64,
    signal_to_noise_ratios: [f64; 3]
) -> [f64; 3] {
    let sum_of_snr: f64 = signal_to_noise_ratios.iter().sum();

    let allocation_a = allocation *
        (signal_to_noise_ratios[0] / sum_of_snr);
    let allocation_b = allocation *
        (signal_to_noise_ratios[1] / sum_of_snr);
    let allocation_c = allocation *
        (signal_to_noise_ratios[2] / sum_of_snr);

    [
        allocation_a,
        allocation_b,
        allocation_c
    ]
}

fn coefficient_of_variance(
    a_mean: f64,
    b_mean: f64,
    covariance_ab: f64,
    a_allocation: f64,
    b_allocation: f64
) -> f64 {
    let expected_return =  a_mean * a_allocation + b_mean * b_allocation;
    let std_dev = (a_allocation.powi(2) + b_allocation.powi(2)
        + (2. * (a_allocation * b_allocation
        * covariance_ab))
    ).sqrt();
    let coefficient = 100. * (std_dev / expected_return);

    coefficient
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
    let rstd = (a_variance.sqrt() / a_mean).abs() * 100.;
    let volatility_index = (rstd / period) * 100.;
    let cv = volatility_index /
        (a_mean * (3_f64).powi(-1)) * 100.;
    let snr = cv.powi(-1);

    [ volatility_index, snr ]
}

fn benchmark_strategies(allocation: f64) {
    let [ lp_supply, volumes, liquidity, torn_eth, eth_usd ] = parse_json_sources();

    let torn_usd = timeseries_eth_denom_to_usd(eth_usd.clone(), torn_eth);
    let liquidity_usd = series_volume_usd(eth_usd.clone(), torn_usd.clone(), liquidity );
    let volumes_usd = series_volume_usd(eth_usd.clone(), torn_usd.clone(), volumes);

    let isolated_torn_usd: Vec<f64> = torn_usd.iter().map(|e| e[1]).collect();
    let isolated_eth_usd: Vec<f64> = eth_usd.iter().map(|e| e[1]).collect();
    let isolated_lp_usd: Vec<f64> = lp_supply.iter().enumerate()
        .map(|(i, e)| (liquidity_usd[i][0] + liquidity_usd[i][1]) / e[1])
        .collect();

    let series_set_size = usize::clone(&torn_usd.len()) as f64;
    let torn_usd_mean = mean(isolated_torn_usd.clone());
    let eth_usd_mean = mean(isolated_eth_usd.clone());
    let lp_usd_mean = mean(isolated_lp_usd.clone());

    let covariance = covariance_matrix(
        isolated_eth_usd,
        isolated_torn_usd,
        isolated_lp_usd,
        eth_usd_mean,
        torn_usd_mean,
        lp_usd_mean
    );

    let [ eth_usd_volatility_index, eth_usd_snr ] = volatility_and_snr(
        covariance[[0, 0, 0]],
        f64::clone(&eth_usd_mean),
        f64::clone(&series_set_size)
    );
    let [ torn_usd_volatility_index, torn_usd_snr ] = volatility_and_snr(
        covariance[[1, 1, 0]],
        f64::clone(&torn_usd_mean),
        f64::clone(&series_set_size)
    );
    let [ lp_usd_volatility_index, lp_usd_snr ] = volatility_and_snr(
        covariance[[2, 2, 0]],
        f64::clone(&lp_usd_mean),
        f64::clone(&series_set_size)
    );

    let [ allocation_a, allocation_b, allocation_c ] = snr_criterion(
        allocation,
        [ eth_usd_snr, torn_usd_snr, lp_usd_snr ]
    );

    println!("ETH");
    println!("VI: {:?}", eth_usd_volatility_index);
    println!("CV: {:?} %", eth_usd_snr.powi(-1));
    println!("S/N: {:?}", eth_usd_snr);

    println!("TORN");
    println!("VI: {:?}", torn_usd_volatility_index);
    println!("CV: {:?} %", torn_usd_snr.powi(-1));
    println!("S/N: {:?}", torn_usd_snr);

    println!("UNIV2-ETH-TORN");
    println!("VI: {:?}", lp_usd_volatility_index);
    println!("CV: {:?} %", lp_usd_snr.powi(-1));
    println!("S/N: {:?}", lp_usd_snr);

    println!("PORTFOLIO");
    println!("${:?} in ETH", allocation_a);
    println!("${:?} in TORN", allocation_b);
    println!("${:?} in UNIV2-ETH-TORN", allocation_c);

}

#[test]
fn check_conditioning() {
    benchmark_strategies(1000000.)
}
