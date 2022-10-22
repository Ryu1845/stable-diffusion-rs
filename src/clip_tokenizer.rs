pub struct SimpleTokenizer {}
impl SimpleTokenizer {
    pub fn new() -> SimpleTokenizer {
        SimpleTokenizer {}
    }
    pub fn encode(&self, text: &String) -> Vec<u16> {
        println!("Encoding {:?}", text);
        vec![1337]
    }
}
