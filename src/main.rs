use dotenv::dotenv;
use std::env;

use llm_chain::chains::{map_reduce, sequential};
use llm_chain::step::Step;
use llm_chain::traits::Executor;
use llm_chain::{parameters, prompt, Parameters};
use llm_chain_openai::chatgpt::PerExecutor;

use tiktoken_rs::cl100k_base;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let openai_key = env::var("OPENAI_API_KEY").unwrap();

    let exec = llm_chain_openai::chatgpt::Executor::new_with_options(
        Some(PerExecutor {
            api_key: Some(openai_key),
        }),
        None,
    )?;

    let map_prompt = Step::for_prompt_template(prompt!(
        "You are a senior software developer. You will review a source code file and its patch",
        "The following is a patch. Please summarize key changes.\n{{text}}"
    ));

    // Create the "reduce" step to combine multiple summaries into one
    let reduce_prompt = Step::for_prompt_template(prompt!(
        "You are a senior software developer. You have reviewed the source code files and its patches, created some interim notes and now create a summary",
        "Please combine the pieces and make a summary:\n{{text}}"
    ));

    let bpe = cl100k_base()?;
    //  Create a map-reduce chain with the map and reduce steps
    let map_chain = map_reduce::Chain::new(map_prompt, reduce_prompt);

    let article = include_str!("article_to_summarize.md");
    // let article = include_str!("article_to_summarize.md");
    // let tokens = bpe.encode_ordinary(&article);
    let tokens = bpe.encode_with_special_tokens(article);
    let chunked = tokens
        .chunks(1800)
        .map(|c| bpe.decode(c.to_vec()).unwrap())
        .collect::<Vec<String>>();

    let docs = chunked.iter().map(|c| parameters!(c)).collect::<Vec<_>>();

    // let docs = vec![parameters!(article)];

    // for d in docs.clone().iter() {
    //     let text = d.get_text().unwrap().to_string();
    //     let head = text.chars().take(50).collect::<String>();
    //     let tokens = bpe.encode_with_special_tokens(&text);

    //     println!("{:?}, {}", head, tokens.len());
    // }
    let res = map_chain.run(docs, Parameters::new(), &exec).await.unwrap();
    println!("{}", res);

    let step = Step::for_prompt_template(prompt!(
        "You're project manager",
        "Report to your superior what you've just done in one sentence:\n{{text}}"
    ));

    let res = step.run(&parameters!(res.to_string()), &exec).await?;
    println!("{}", res);

    Ok(())
}
