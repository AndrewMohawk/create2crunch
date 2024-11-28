#![warn(unused_crate_dependencies, unreachable_pub)]
#![deny(unused_must_use, rust_2018_idioms)]

use alloy_primitives::{hex, Address, FixedBytes};
use byteorder::{BigEndian, ByteOrder, LittleEndian};
use console::Term;
use fs4::FileExt;
use ocl::{Buffer, Context, Device, MemFlags, Platform, ProQue, Program, Queue};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use separator::Separatable;
use std::error::Error;
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};
use terminal_size::{terminal_size, Height};
use tiny_keccak::{Hasher, Keccak};

use crate::scoring::score_address_hex;
mod gpu_config;
mod patterns;
mod scoring;

const WORK_SIZE: u32 = 0x80000000;
const WORK_FACTOR: u128 = (WORK_SIZE as u128) / 1_000_000;
const CONTROL_CHARACTER: u8 = 0xff;
const MAX_INCREMENTER: u64 = 0xffffffffffff;

static KERNEL_SRC: &str = include_str!("./kernels/keccak256.cl");

pub struct Config {
    pub factory_address: [u8; 20],
    pub calling_address: [u8; 20],
    pub init_code_hash: [u8; 32],
    pub gpu_device: u8,
    pub leading_zeroes_threshold: u8,
    pub total_zeroes_threshold: u8,
}

impl Config {
    pub fn new(mut args: std::env::Args) -> Result<Self, &'static str> {
        args.next();

        let Some(factory_address_string) = args.next() else {
            return Err("didn't get a factory_address argument");
        };
        let Some(calling_address_string) = args.next() else {
            return Err("didn't get a calling_address argument");
        };
        let Some(init_code_hash_string) = args.next() else {
            return Err("didn't get an init_code_hash argument");
        };

        let gpu_device_string = match args.next() {
            Some(arg) => arg,
            None => String::from("255"),
        };
        let leading_zeroes_threshold_string = match args.next() {
            Some(arg) => arg,
            None => String::from("8"),  // Increased minimum leading zeros
        };
        let total_zeroes_threshold_string = match args.next() {
            Some(arg) => arg,
            None => String::from("8"),  // Increased total zeros threshold
        };

        let Ok(factory_address_vec) = hex::decode(factory_address_string) else {
            return Err("could not decode factory address argument");
        };
        let Ok(calling_address_vec) = hex::decode(calling_address_string) else {
            return Err("could not decode calling address argument");
        };
        let Ok(init_code_hash_vec) = hex::decode(init_code_hash_string) else {
            return Err("could not decode initialization code hash argument");
        };

        let Ok(factory_address) = factory_address_vec.try_into() else {
            return Err("invalid length for factory address argument");
        };
        let Ok(calling_address) = calling_address_vec.try_into() else {
            return Err("invalid length for calling address argument");
        };
        let Ok(init_code_hash) = init_code_hash_vec.try_into() else {
            return Err("invalid length for initialization code hash argument");
        };

        let Ok(gpu_device) = gpu_device_string.parse::<u8>() else {
            return Err("invalid gpu device value");
        };
        let Ok(leading_zeroes_threshold) = leading_zeroes_threshold_string.parse::<u8>() else {
            return Err("invalid leading zeroes threshold value supplied");
        };
        let Ok(total_zeroes_threshold) = total_zeroes_threshold_string.parse::<u8>() else {
            return Err("invalid total zeroes threshold value supplied");
        };

        if leading_zeroes_threshold > 20 {
            return Err("invalid value for leading zeroes threshold argument. (valid: 0..=20)");
        }
        if total_zeroes_threshold > 20 && total_zeroes_threshold != 255 {
            return Err("invalid value for total zeroes threshold argument. (valid: 0..=20 | 255)");
        }

        Ok(Self {
            factory_address,
            calling_address,
            init_code_hash,
            gpu_device,
            leading_zeroes_threshold,
            total_zeroes_threshold,
        })
    }
}

pub fn cpu(config: Config) -> Result<(), Box<dyn Error>> {
    let file = output_file();

    loop {
        let mut header = [0; 47];
        header[0] = CONTROL_CHARACTER;
        header[1..21].copy_from_slice(&config.factory_address);
        header[21..41].copy_from_slice(&config.calling_address);
        header[41..].copy_from_slice(&FixedBytes::<6>::random()[..]);

        let mut hash_header = Keccak::v256();
        hash_header.update(&header);

        (0..MAX_INCREMENTER)
            .into_par_iter()
            .for_each(|salt| {
                let salt = salt.to_le_bytes();
                let salt_incremented_segment = &salt[..6];

                let mut hash = hash_header.clone();
                hash.update(salt_incremented_segment);
                hash.update(&config.init_code_hash);

                let mut res: [u8; 32] = [0; 32];
                hash.finalize(&mut res);

                let address = <&Address>::try_from(&res[12..]).unwrap();
                let address_hex = format!("{address}");

                if let Some(score) = score_address_hex(&address_hex) {
                    if score > 174 {  // Only keep competitive scores
                        let header_hex_string = hex::encode(header);
                        let body_hex_string = hex::encode(salt_incremented_segment);
                        let full_salt = format!("0x{}{}", &header_hex_string[42..], &body_hex_string);

                        let output = format!(
                            "{full_salt} => {address} => {score}"
                        );
                        println!("{output}");

                        file.lock_exclusive().expect("Couldn't lock file.");
                        writeln!(&file, "{output}")
                            .expect("Couldn't write to `efficient_addresses.txt` file.");
                        file.unlock().expect("Couldn't unlock file.");
                    }
                }
            });
    }
}

pub fn gpu(config: Config) -> ocl::Result<()> {
    let work_group_size = gpu_config::get_optimal_work_group_size();
    let optimal_patterns = patterns::get_optimal_pattern_sequence();
    
    println!(
        "Setting up experimental OpenCL miner using device {}...",
        config.gpu_device
    );
    println!("Starting pattern-based mining...");
    let mut pattern_index: usize = 0;

    let file = output_file();
    let mut found: u64 = 0;
    let mut found_list: Vec<String> = vec![];
    let term = Term::stdout();
    let platform = Platform::new(ocl::core::default_platform()?);
    let device = Device::by_idx_wrap(platform, config.gpu_device as usize)?;
    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;
    
    // Configure OpenCL for H100 and use it
    let program = Program::builder()
        .devices(device)
        .src(mk_kernel_src(&config))
        .cmplr_opt(&format!(
            "-cl-std=CL2.0 -D WORKGROUP_SIZE={} -D TOTAL_ZEROES={} -D LEADING_ZEROES={} \
             -cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros \
             -cl-denorms-are-zero -cl-single-precision-constant",
            work_group_size,
            config.total_zeroes_threshold,
            config.leading_zeroes_threshold
        ))
        .build(&context)?;

    let queue = Queue::new(&context, device, None)?;
    let ocl_pq = ProQue::new(context, queue, program, Some(WORK_SIZE / 16));
    let mut rng = thread_rng();
    let start_time: f64 = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    let mut rate: f64 = 0.0;
    let mut cumulative_nonce: u64 = 0;
    let mut previous_time: f64 = 0.0;
    let mut work_duration_millis: u64 = 0;

    loop {
        let current_pattern = optimal_patterns[pattern_index];
        pattern_index = (pattern_index + 1) % optimal_patterns.len();

        let mut pattern = [0u8; 4];
        pattern.copy_from_slice(&current_pattern);
        let salt = FixedBytes::<4>::try_from(&pattern[..]).unwrap();

        let message_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().read_only())
            .len(4)
            .copy_host_slice(&salt[..])
            .build()?;

        let mut nonce: [u32; 1] = rng.gen();
        let mut view_buf = [0; 8];
        let mut nonce_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().read_only())
            .len(1)
            .copy_host_slice(&nonce)
            .build()?;

        let mut solutions: Vec<u64> = vec![0; 1];
        let solutions_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().write_only())
            .len(1)
            .copy_host_slice(&solutions)
            .build()?;

        loop {
            let kern = ocl_pq
                .kernel_builder("hashMessage")
                .arg_named("message", None::<&Buffer<u8>>)
                .arg_named("nonce", None::<&Buffer<u32>>)
                .arg_named("solutions", None::<&Buffer<u64>>)
                .build()?;

            kern.set_arg("message", Some(&message_buffer))?;
            kern.set_arg("nonce", Some(&nonce_buffer))?;
            kern.set_arg("solutions", &solutions_buffer)?;

            unsafe { kern.enq()? };

            let mut now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
            let current_time = now.as_secs() as f64;
            let print_output = current_time - previous_time > 0.99;
            previous_time = current_time;

            if print_output {
                term.clear_screen()?;
                let total_runtime = current_time - start_time;
                let total_runtime_hrs = total_runtime as u64 / 3600;
                let total_runtime_mins = (total_runtime as u64 - total_runtime_hrs * 3600) / 60;
                let total_runtime_secs = total_runtime
                    - (total_runtime_hrs * 3600) as f64
                    - (total_runtime_mins * 60) as f64;

                let work_rate: u128 = WORK_FACTOR * cumulative_nonce as u128;
                if total_runtime > 0.0 {
                    rate = 1.0 / total_runtime;
                }

                LittleEndian::write_u64(&mut view_buf, (nonce[0] as u64) << 32);
                let height = terminal_size().map(|(_w, Height(h))| h).unwrap_or(10);

                term.write_line(&format!(
                    "total runtime: {}:{:02}:{:02} ({} cycles)\t\t\t\
                     work size per cycle: {}",
                    total_runtime_hrs,
                    total_runtime_mins,
                    total_runtime_secs,
                    cumulative_nonce,
                    WORK_SIZE.separated_string(),
                ))?;

                term.write_line(&format!(
                    "rate: {:.2} million attempts per second\t\t\t\
                     total found this run: {}",
                    work_rate as f64 * rate,
                    found
                ))?;

                term.write_line(&format!(
                    "current search space: {}xxxxxxxx{:08x}\t\t\
                     threshold: {} leading or {} total zeroes",
                    hex::encode(salt),
                    BigEndian::read_u64(&view_buf),
                    config.leading_zeroes_threshold,
                    config.total_zeroes_threshold
                ))?;

                let rows = if height < 5 { 1 } else { height as usize - 4 };
                let last_rows: Vec<String> = found_list.iter().cloned().rev().take(rows).collect();
                let ordered: Vec<String> = last_rows.iter().cloned().rev().collect();
                let recently_found = &ordered.join("\n");
                term.write_line(recently_found)?;
            }

            cumulative_nonce += 1;
            let work_start_time_millis = now.as_secs() * 1000 + now.subsec_nanos() as u64 / 1000000;

            if work_duration_millis != 0 {
                std::thread::sleep(std::time::Duration::from_millis(
                    work_duration_millis * 980 / 1000,
                ));
            }

            solutions_buffer.read(&mut solutions).enq()?;

            now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
            work_duration_millis = (now.as_secs() * 1000 + now.subsec_nanos() as u64 / 1000000)
                - work_start_time_millis;

            if solutions[0] != 0 {
                break;
            }

            nonce[0] += 1;
            nonce_buffer = Buffer::builder()
                .queue(ocl_pq.queue().clone())
                .flags(MemFlags::new().read_write())
                .len(1)
                .copy_host_slice(&nonce)
                .build()?;
        }

        for &solution in &solutions {
            if solution == 0 {
                continue;
            }

            let solution = solution.to_le_bytes();

            let mut solution_message = [0; 85];
            solution_message[0] = CONTROL_CHARACTER;
            solution_message[1..21].copy_from_slice(&config.factory_address);
            solution_message[21..41].copy_from_slice(&config.calling_address);
            solution_message[41..45].copy_from_slice(&salt[..]);
            solution_message[45..53].copy_from_slice(&solution);
            solution_message[53..].copy_from_slice(&config.init_code_hash);

            let mut hash = Keccak::v256();
            hash.update(&solution_message);

            let mut res: [u8; 32] = [0; 32];
            hash.finalize(&mut res);

            let address = <&Address>::try_from(&res[12..]).unwrap();
            let address_hex = format!("{}", address);

            if let Some(score) = score_address_hex(&address_hex) {
                if score > 174 {
                    found += 1;
                    let output = format!(
                        "0x{}{}{} => {} => {}",
                        hex::encode(config.calling_address),
                        hex::encode(salt),
                        hex::encode(solution),
                        address,
                        score
                    );

                    let show = format!("{output} (Score: {score})");
                    found_list.push(show.to_string());

                    file.lock_exclusive().expect("Couldn't lock file.");
                    writeln!(&file, "{output}")
                        .expect("Couldn't write to `efficient_addresses.txt` file.");
                    file.unlock().expect("Couldn't unlock file.");
                }
            }
        }
    }
}

#[track_caller]
fn output_file() -> File {
    OpenOptions::new()
        .append(true)
        .create(true)
        .read(true)
        .open("efficient_addresses.txt")
        .expect("Could not create or open `efficient_addresses.txt` file.")
}

fn get_score_check_code() -> String {
    r#"
    bool check_score(uchar* addr, int addr_len) {
        int score = 0;
        bool found_first = false;
        int first_non_zero = -1;
        
        // Count leading zeros and check first non-zero is 4
        for(int i = 0; i < addr_len * 2; i++) {
            uchar nibble = (i % 2 == 0) ? (addr[i/2] >> 4) : (addr[i/2] & 0xF);
            if(!found_first) {
                if(nibble == 0) {
                    score += 10;
                } else {
                    found_first = true;
                    first_non_zero = i;
                    if(nibble != 4) return false;
                }
            }
            
            // Count total 4s
            if(nibble == 4) {
                score += 1;
            }
        }
        
        // Look for 4444 sequence
        bool found_four_fours = false;
        for(int i = 0; i < addr_len * 2 - 3; i++) {
            uchar n1 = (i % 2 == 0) ? (addr[i/2] >> 4) : (addr[i/2] & 0xF);
            uchar n2 = ((i+1) % 2 == 0) ? (addr[(i+1)/2] >> 4) : (addr[(i+1)/2] & 0xF);
            uchar n3 = ((i+2) % 2 == 0) ? (addr[(i+2)/2] >> 4) : (addr[(i+2)/2] & 0xF);
            uchar n4 = ((i+3) % 2 == 0) ? (addr[(i+3)/2] >> 4) : (addr[(i+3)/2] & 0xF);
            
            if(n1 == 4 && n2 == 4 && n3 == 4 && n4 == 4) {
                score += 40;
                if(i + 4 < addr_len * 2) {
                    uchar next = ((i+4) % 2 == 0) ? (addr[(i+4)/2] >> 4) : (addr[(i+4)/2] & 0xF);
                    if(next != 4) score += 20;
                }
                found_four_fours = true;
                break;
            }
        }

        // Check last 4 nibbles for 4444
        int last_start = addr_len * 2 - 4;
        if(last_start >= 0) {
            bool last_four = true;
            for(int i = 0; i < 4; i++) {
                uchar n = ((last_start + i) % 2 == 0) ? 
                    (addr[(last_start + i)/2] >> 4) : 
                    (addr[(last_start + i)/2] & 0xF);
                if(n != 4) {
                    last_four = false;
                    break;
                }
            }
            if(last_four) score += 20;
        }
        
        return score > 174;  // Only keep competitive scores
    }
    "#.to_string()
}

fn mk_kernel_src(config: &Config) -> String {
    let mut src = String::with_capacity(2048 + KERNEL_SRC.len());
    
    let factory = config.factory_address.iter();
    let caller = config.calling_address.iter();
    let hash = config.init_code_hash.iter();
    
    // Generate array definitions for the constants
    writeln!(src, "__constant uchar factory_address[20] = {{").unwrap();
    for (i, x) in factory.enumerate() {
        if i > 0 { write!(src, ", ").unwrap(); }
        write!(src, "0x{:02x}u", x).unwrap();
    }
    writeln!(src, "}};").unwrap();
    
    writeln!(src, "__constant uchar caller_address[20] = {{").unwrap();
    for (i, x) in caller.enumerate() {
        if i > 0 { write!(src, ", ").unwrap(); }
        write!(src, "0x{:02x}u", x).unwrap();
    }
    writeln!(src, "}};").unwrap();
    
    writeln!(src, "__constant uchar init_code_hash[32] = {{").unwrap();
    for (i, x) in hash.enumerate() {
        if i > 0 { write!(src, ", ").unwrap(); }
        write!(src, "0x{:02x}u", x).unwrap();
    }
    writeln!(src, "}};").unwrap();
    
    let lz = config.leading_zeroes_threshold;
    writeln!(src, "#define LEADING_ZEROES {lz}").unwrap();
    let tz = config.total_zeroes_threshold;
    writeln!(src, "#define TOTAL_ZEROES {tz}").unwrap();

    src.push_str(&get_score_check_code());
    src.push_str(KERNEL_SRC);

    src
}
