#![warn(unused_crate_dependencies, unreachable_pub)]
#![deny(unused_must_use, rust_2018_idioms)]

use alloy_primitives::{hex, FixedBytes};
use byteorder::{LittleEndian, WriteBytesExt};
use console::Term;
use ocl::{Buffer, Context, Device, Kernel, MemFlags, Platform, Program, Queue};
use rand::{thread_rng, Rng};
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use tiny_keccak::{Hasher, Keccak};

mod reward;
pub use reward::Reward;

const WORK_SIZE: u32 = 0x80000000;
const WORKGROUP_SIZE: u32 = 256;
const NUM_PARALLEL_BUFFERS: usize = 4;

/* Server */
// const WORK_SIZE: u32 = 0xD0000000;
// const WORKGROUP_SIZE: u32 = 1024;
// const NUM_PARALLEL_BUFFERS: usize = 256;

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
            None => String::from("3"),
        };
        let total_zeroes_threshold_string = match args.next() {
            Some(arg) => arg,
            None => String::from("5"),
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

struct BufferSet {
    message: Buffer<u8>,
    nonce: Buffer<u32>,
    results: Buffer<u32>,
}

pub fn gpu(config: Config) -> ocl::Result<()> {
    println!(
        "Setting up experimental OpenCL miner using device {}...",
        config.gpu_device
    );

    let _file = output_file();
    let _rewards = Reward::new();
    let mut _found: u64 = 0;
    let mut _found_list: Vec<String> = vec![];
    let _term = Term::stdout();

    let platform = Platform::new(ocl::core::default_platform()?);
    let device = Device::by_idx_wrap(platform, config.gpu_device as usize)?;
    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;

    let queues: Vec<Queue> = (0..NUM_PARALLEL_BUFFERS)
        .map(|_| Queue::new(&context, device, None))
        .collect::<ocl::Result<_>>()?;

    let program = Program::builder()
        .devices(device)
        .src(mk_kernel_src(&config))
        .build(&context)?;

    let mut rng = thread_rng();
    let mut highest_score = 0;

    let mut buffer_sets: Vec<BufferSet> = Vec::with_capacity(NUM_PARALLEL_BUFFERS);
    let results_size = 6 * (WORK_SIZE as usize / NUM_PARALLEL_BUFFERS);

    // Create buffer sets with optimized memory flags
    for queue in &queues {
        let salt = FixedBytes::<4>::random();
        let nonce_init = rng.gen::<u32>();

        let message_buffer = Buffer::builder()
            .queue(queue.clone())
            .flags(MemFlags::new().read_only())
            .len(4)
            .copy_host_slice(&salt[..])
            .build()?;

        let nonce_buffer = Buffer::builder()
            .queue(queue.clone())
            .flags(MemFlags::new().read_only())
            .len(1)
            .copy_host_slice(&[nonce_init])
            .build()?;

        let results_buffer = Buffer::builder()
            .queue(queue.clone())
            .flags(MemFlags::new().write_only())
            .len(results_size)
            .build()?;

        buffer_sets.push(BufferSet {
            message: message_buffer,
            nonce: nonce_buffer,
            results: results_buffer,
        });
    }

    let kernels: Vec<Kernel> = buffer_sets
        .iter()
        .zip(&queues)
        .map(|(buffer_set, queue)| {
            ocl::Kernel::builder()
                .program(&program)
                .name("hashMessage")
                .queue(queue.clone())
                .arg(&buffer_set.message)
                .arg(&buffer_set.nonce)
                .arg(&buffer_set.results)
                .global_work_size(WORK_SIZE / NUM_PARALLEL_BUFFERS as u32)
                .local_work_size(WORKGROUP_SIZE)
                .build()
        })
        .collect::<ocl::Result<_>>()?;

    let mut results = vec![0u32; results_size];

    loop {
        // Launch kernels asynchronously
        for kernel in &kernels {
            unsafe {
                kernel
                    .cmd()
                    .queue(&kernel.default_queue().unwrap())
                    .global_work_size(WORK_SIZE / NUM_PARALLEL_BUFFERS as u32)
                    .local_work_size(WORKGROUP_SIZE)
                    .enq()?;
            }
        }

        // Process results and update nonces
        for buffer_set in buffer_sets.iter() {
            // Read results and ensure completion
            buffer_set.results.read(&mut results).enq()?;
            buffer_set.results.default_queue().unwrap().finish()?;

            // Process results
            for i in 0..results_size / 6 {
                let idx = i * 6;
                let score = results[idx];
                if score > highest_score {
                    highest_score = score;
                    process_result(&config, &results[idx..idx + 6], score, buffer_set, i as u32)?;
                }
            }

            // Update nonce for next iteration
            let mut nonce_vec = vec![0u32; 1];
            buffer_set.nonce.read(&mut nonce_vec).enq()?;
            buffer_set.nonce.default_queue().unwrap().finish()?;

            nonce_vec[0] += WORK_SIZE / NUM_PARALLEL_BUFFERS as u32;

            buffer_set.nonce.write(&nonce_vec).enq()?;
            buffer_set.nonce.default_queue().unwrap().finish()?;
        }
    }
}

fn process_result(
    config: &Config,
    result_slice: &[u32],
    score: u32,
    buffer_set: &BufferSet,
    work_idx: u32,
) -> ocl::Result<()> {
    let mut address_bytes = [0u8; 20];
    for j in 0..5 {
        let val = result_slice[j + 1];
        address_bytes[4 * j] = (val >> 24) as u8;
        address_bytes[4 * j + 1] = (val >> 16) as u8;
        address_bytes[4 * j + 2] = (val >> 8) as u8;
        address_bytes[4 * j + 3] = val as u8;
    }

    let address_hex = hex::encode(address_bytes);

    let mut nonce_vec = vec![0u32; 1];
    buffer_set.nonce.read(&mut nonce_vec).enq()?;

    let nonce_uint32_t = [work_idx, nonce_vec[0]];
    let mut nonce_bytes = [0u8; 8];
    {
        let mut cursor = std::io::Cursor::new(&mut nonce_bytes[..]);
        cursor.write_u32::<LittleEndian>(nonce_uint32_t[0])?;
        cursor.write_u32::<LittleEndian>(nonce_uint32_t[1])?;
    }

    let mut message_vec = vec![0u8; 4];
    buffer_set.message.read(&mut message_vec).enq()?;

    let full_salt = [
        &config.calling_address[..],
        &message_vec[..],
        &nonce_bytes[..],
    ]
    .concat();

    // Verify the address
    let mut data = Vec::with_capacity(1 + 20 + 32 + 32);
    data.push(0xffu8);
    data.extend_from_slice(&config.factory_address);
    data.extend_from_slice(&full_salt);
    data.extend_from_slice(&config.init_code_hash);

    let mut hasher = Keccak::v256();
    hasher.update(&data);
    let mut address_hash = [0u8; 32];
    hasher.finalize(&mut address_hash);

    let address_hex_from_hash = hex::encode(&address_hash[12..]);

    if address_hex_from_hash != address_hex {
        println!(
            "Address mismatch! Computed: {}, Expected: {}",
            address_hex_from_hash, address_hex
        );
    } else {
        println!("Address verified successfully.");
    }

    let output = format!(
        "0x{} => {} => {}",
        hex::encode(full_salt),
        address_hex,
        score,
    );
    println!("{}", output);

    Ok(())
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

fn mk_kernel_src(config: &Config) -> String {
    let mut src = String::with_capacity(2048 + KERNEL_SRC.len());

    let factory = config.factory_address.iter();
    let caller = config.calling_address.iter();
    let hash = config.init_code_hash.iter();
    let hash = hash.enumerate().map(|(i, x)| (i + 52, x));
    for (i, x) in factory.chain(caller).enumerate().chain(hash) {
        writeln!(src, "#define S_{} {}u", i + 1, x).unwrap();
    }

    let lz = config.leading_zeroes_threshold;
    writeln!(src, "#define LEADING_ZEROES {lz}").unwrap();
    let tz = config.total_zeroes_threshold;
    writeln!(src, "#define TOTAL_ZEROES {tz}").unwrap();
    writeln!(src, "#define LOCAL_SIZE {}", WORKGROUP_SIZE).unwrap();

    src.push_str(KERNEL_SRC);
    src
}
