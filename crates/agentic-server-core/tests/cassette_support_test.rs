mod support;

#[test]
fn cassette_accepts_structured_responses_input() {
    let cassette = serde_yaml::from_str::<support::Cassette>(
        r"
turns:
- request:
    path: /v1/responses
    body:
      input:
      - role: user
        content: hello
  response:
    body: {}
",
    );

    assert!(
        cassette.is_ok(),
        "structured Responses input should deserialize: {cassette:?}"
    );
}

#[test]
fn request_builder_preserves_structured_responses_input() {
    let input = serde_json::json!([
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe this image."},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,abc",
                    "detail": "low"
                }
            ]
        }
    ]);

    let request = support::make_request(&input, false, false, None, None);

    let serialized = serde_json::to_value(request.input).expect("serialize request input");
    assert_eq!(serialized[0]["role"], "user");
    assert_eq!(serialized[0]["content"][0]["text"], "Describe this image.");
    assert_eq!(serialized[0]["content"][1]["type"], "input_image");
    assert_eq!(serialized[0]["content"][1]["image_url"], "data:image/png;base64,abc");
    assert_eq!(serialized[0]["content"][1]["detail"], "low");
}
