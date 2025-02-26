use clap::{Command, Arg};

fn main() {
    let matches = Command::new("run")
        .version("1.0")
        .about("Runs the model with the specified options")
        .override_usage("run <checkpoint> [options]")
        .after_help("Example: run model.bin -n 256 -i \"Once upon a time\"")
        .arg(Arg::new("checkpoint")
            .help("Path to the checkpoint file")
            .required(true)
            .index(1)
            .value_parser(clap::value_parser!(String)))
        .arg(Arg::new("temperature")
            .short('t')
            .long("temperature")
            .value_name("float")
            .help("temperature in [0,inf], default 1.0")
            .value_parser(clap::value_parser!(f32)))
        .arg(Arg::new("p_value")
            .short('p')
            .long("p_value")
            .value_name("float")
            .help("p value in top-p (nucleus) sampling in [0,1], default 0.9")
            .value_parser(clap::value_parser!(f32)))
        .arg(Arg::new("seed")
            .short('s')
            .long("seed")
            .value_name("int")
            .help("random seed, default time(NULL)")
            .value_parser(clap::value_parser!(u64)))
        .arg(Arg::new("steps")
            .short('n')
            .long("steps")
            .value_name("int")
            .help("number of steps to run for, default 256. 0 = max_seq_len")
            .value_parser(clap::value_parser!(i32)))
        .arg(Arg::new("input_prompt")
            .short('i')
            .long("input_prompt")
            .value_name("string")
            .help("input prompt")
            .value_parser(clap::value_parser!(String)))
        .arg(Arg::new("tokenizer")
            .short('z')
            .long("tokenizer")
            .value_name("string")
            .help("optional path to custom tokenizer")
            .value_parser(clap::value_parser!(String)))
        .arg(Arg::new("mode")
            .short('m')
            .long("mode")
            .value_name("string")
            .help("mode: generate|chat, default: generate")
            .value_parser(clap::value_parser!(String)))
        .arg(Arg::new("system_prompt")
            .short('y')
            .long("system_prompt")
            .value_name("string")
            .help("(optional) system prompt in chat mode")
            .value_parser(clap::value_parser!(String)))
        .get_matches();

    let checkpoint = matches.get_one::<String>("checkpoint").unwrap();
    let temperature = matches.get_one::<f32>("temperature").copied().unwrap_or(1.0);
    let p_value = matches.get_one::<f32>("p_value").copied().unwrap_or(0.9);
    let seed = matches.get_one::<u64>("seed").copied().unwrap_or(0);
    let steps = matches.get_one::<i32>("steps").copied().unwrap_or(256);
    let input_prompt = matches.get_one::<String>("input_prompt").map(|s| s.as_str()).unwrap_or("");
    let tokenizer = matches.get_one::<String>("tokenizer").map(|s| s.as_str()).unwrap_or("tokenizer.bin");
    let mode = matches.get_one::<String>("mode").map(|s| s.as_str()).unwrap_or("generate");
    let system_prompt = matches.get_one::<String>("system_prompt").map(|s| s.as_str()).unwrap_or("");

    println!("Checkpoint: {}", checkpoint);
    println!("Temperature: {}", temperature);
    println!("P value: {}", p_value);
    println!("Seed: {}", seed);
    println!("Steps: {}", steps);
    println!("Input prompt: {}", input_prompt);
    println!("Tokenizer: {}", tokenizer);
    println!("Mode: {}", mode);
    println!("System prompt: {}", system_prompt);

    // Your code to run the model with the specified options goes here
}
