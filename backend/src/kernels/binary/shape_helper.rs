#[derive(Debug, Clone)]
pub struct BroadcastShapeHelper {
    pub out_shape: Vec<i64>,
    pub a_padded: Vec<i64>,
    pub b_padded: Vec<i64>,
    pub needs_boardcast: bool,
    pub simple_boardcast: bool,
}

impl BroadcastShapeHelper {
    pub fn new(mut a_shape: Vec<i64>, mut b_shape: Vec<i64>) -> Result<Self, &'static str> {
        let mut needs_boardcast = false;
        let mut simple_boardcast = true;

        if a_shape.is_empty() && b_shape.is_empty() {
            return Ok(Self {
                a_padded: a_shape,
                b_padded: b_shape,
                out_shape: Vec::new(),
                needs_boardcast: false,
                simple_boardcast: false,
            });
        }

        match a_shape.len().cmp(&b_shape.len()) {
            std::cmp::Ordering::Greater => {
                b_shape.reverse();
                b_shape.resize_with(a_shape.len(), || 1);
                b_shape.reverse();

                needs_boardcast = true;
            }
            std::cmp::Ordering::Less => {
                a_shape.reverse();
                a_shape.resize_with(b_shape.len(), || 1);
                a_shape.reverse();

                needs_boardcast = true;
            }
            _ => {
                for (a, b) in a_shape.iter().zip(b_shape.iter()) {
                    if a != b {
                        needs_boardcast = true;
                        break;
                    }
                }
            }
        }

        let mut out_shape = a_shape.clone();
        let mut a_leading_ones: usize = 0;
        let mut a_prev_not_one = false;
        let mut b_leading_ones: usize = 0;
        let mut b_prev_not_one = false;
        if needs_boardcast {
            for (indx, (a, b)) in a_shape.iter().zip(b_shape.iter()).enumerate() {
                if *a != *b {
                    if *a == 1 {
                        out_shape[indx] = *b;

                        if *b != 1 {
                            b_prev_not_one = true;
                        }
                        if !a_prev_not_one {
                            a_leading_ones += 1;
                        }
                    } else if *b == 1 {
                        //no need to set sense out_shape starts as a_shape
                        if !b_prev_not_one {
                            b_leading_ones += 1;
                        }
                        if *a != 1 {
                            a_prev_not_one = true;
                        }
                    } else {
                        return Err("Shapes cannot be broadcast");
                    }
                } else if *a == 1 && *b == 1 {
                    if !b_prev_not_one {
                        b_leading_ones += 1;
                    }
                    if !a_prev_not_one {
                        a_leading_ones += 1;
                    }
                } else {
                    a_prev_not_one = true;
                    b_prev_not_one = true;
                }
            }
        }

        if a_leading_ones < b_leading_ones {
            for i in b_leading_ones..out_shape.len() {
                if a_shape[i] != b_shape[i] {
                    simple_boardcast = false;
                }
            }
        } else {
            for i in a_leading_ones..out_shape.len() {
                if a_shape[i] != b_shape[i] {
                    simple_boardcast = false;
                }
            }
        }

        if needs_boardcast && !simple_boardcast && a_shape.len() > 9 {
            return Err("Max broadcast of 9");
        }

        Ok(Self {
            out_shape,
            a_padded: a_shape,
            b_padded: b_shape,
            needs_boardcast,
            simple_boardcast: simple_boardcast && needs_boardcast,
        })
    }
}
