gert::InfershapeContextPara infershapeContextPara("SortWithIndex",
    { /* inputs */ },
    { /* outputs */ },
    {
        {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
        {"descending", Ops::Math::AnyValue::CreateFrom<bool>(true)},
    });
