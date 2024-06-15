# 대분류 Asepct Category 매핑 Dictionary
# value (list) 내에 포함된 label을 key 값으로 변경
label_map_dict = {
    "만족도": ["만족도","제품만족도","가격","서비스만족도","발송","배송","환불","포장","AS","수리","환불","교환","문의"],
    "키": ["키","키감","소리","키압","통울림","키캡","키축","스테빌라이저","윤활"],
    "디자인": ["디자인","색상","재질","백라이트","외형","키보드각인"],
    "휴대성": ["휴대성","무게","사이즈"],
    "제품구성": ["제품구성","파우치","거치대","키스킨","건전지","설명서","키캡리무버","키보드커버","케이블"],
    "제품품질": ["제품품질","마감","내구성"],
    "성능": ["성능","배터리","수명","호환성","지연율","소비전력"],
    "편의성": ["편의성","손목","키배열","키보드높이"],
    "연결방식": ["연결방식","동글","블루투스","유선"],
    "기능": ["기능","다중페어링","접이식","래피드트리거"]
}


# 대분류 Asepct Category Dictionary에 BIO tag 적용
label_list,label_changing_rule = [], {}
for key in label_map_dict.keys():
    if key != 'O':
        label_list.extend(['B-' + key, 'I-' + key])
    else:
        label_list.append('O')
for key, labels in label_map_dict.items():
    for label in labels:
        if key != label:
            for tag in ["B-", "I-"]:
                label_changing_rule[tag + label] = tag + key