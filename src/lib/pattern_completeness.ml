(****************************************************************************)
(*     Sail                                                                 *)
(*                                                                          *)
(*  Sail and the Sail architecture models here, comprising all files and    *)
(*  directories except the ASL-derived Sail code in the aarch64 directory,  *)
(*  are subject to the BSD two-clause licence below.                        *)
(*                                                                          *)
(*  The ASL derived parts of the ARMv8.3 specification in                   *)
(*  aarch64/no_vector and aarch64/full are copyright ARM Ltd.               *)
(*                                                                          *)
(*  Copyright (c) 2013-2021                                                 *)
(*    Kathyrn Gray                                                          *)
(*    Shaked Flur                                                           *)
(*    Stephen Kell                                                          *)
(*    Gabriel Kerneis                                                       *)
(*    Robert Norton-Wright                                                  *)
(*    Christopher Pulte                                                     *)
(*    Peter Sewell                                                          *)
(*    Alasdair Armstrong                                                    *)
(*    Brian Campbell                                                        *)
(*    Thomas Bauereiss                                                      *)
(*    Anthony Fox                                                           *)
(*    Jon French                                                            *)
(*    Dominic Mulligan                                                      *)
(*    Stephen Kell                                                          *)
(*    Mark Wassell                                                          *)
(*    Alastair Reid (Arm Ltd)                                               *)
(*                                                                          *)
(*  All rights reserved.                                                    *)
(*                                                                          *)
(*  This work was partially supported by EPSRC grant EP/K008528/1 <a        *)
(*  href="http://www.cl.cam.ac.uk/users/pes20/rems">REMS: Rigorous          *)
(*  Engineering for Mainstream Systems</a>, an ARM iCASE award, EPSRC IAA   *)
(*  KTF funding, and donations from Arm.  This project has received         *)
(*  funding from the European Research Council (ERC) under the European     *)
(*  Union’s Horizon 2020 research and innovation programme (grant           *)
(*  agreement No 789108, ELVER).                                            *)
(*                                                                          *)
(*  This software was developed by SRI International and the University of  *)
(*  Cambridge Computer Laboratory (Department of Computer Science and       *)
(*  Technology) under DARPA/AFRL contracts FA8650-18-C-7809 ("CIFV")        *)
(*  and FA8750-10-C-0237 ("CTSRD").                                         *)
(*                                                                          *)
(*  Redistribution and use in source and binary forms, with or without      *)
(*  modification, are permitted provided that the following conditions      *)
(*  are met:                                                                *)
(*  1. Redistributions of source code must retain the above copyright       *)
(*     notice, this list of conditions and the following disclaimer.        *)
(*  2. Redistributions in binary form must reproduce the above copyright    *)
(*     notice, this list of conditions and the following disclaimer in      *)
(*     the documentation and/or other materials provided with the           *)
(*     distribution.                                                        *)
(*                                                                          *)
(*  THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS''      *)
(*  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED       *)
(*  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A         *)
(*  PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR     *)
(*  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,            *)
(*  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT        *)
(*  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF        *)
(*  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND     *)
(*  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,      *)
(*  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT      *)
(*  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF      *)
(*  SUCH DAMAGE.                                                            *)
(****************************************************************************)

open Ast
open Ast_util
module Big_int = Nat_big_num

module IntSet = Util.IntSet
module IntIntSet = Util.IntIntSet
               
type ctx = {
    variants : (typquant * type_union list) Bindings.t;
    enums : IdSet.t Bindings.t;
  }

module type Config =
  sig
    type t
    val typ_of_t : t -> typ
  end

type row_index = {
    loc: l;
    num: int
  }
  
type 'a rows = Rows of ((row_index * 'a) list)
type 'a columns = Columns of ('a list)

type column_type = Tuple_column of int | App_column of id | Bool_column | Enum_column of id | Lit_column | Unknown_column

type complete_info = {
    (* As we check completeness, we check submatrices which correspond to a subset of rows in the overall case statement *)
    rows: IntSet.t;
    (* These literal patterns can be turned into wildcards, as row index number * pattern number pairs *)
    wildcards: IntIntSet.t;
    (* Wildcards we must keep because they cannot be removed in all submatrices *)
    preserved_literals: IntIntSet.t;
    (* These rows are redundant *)
    redundant: IntSet.t;
  }

let union_complete lhs rhs =
  let all_wildcards = IntIntSet.union lhs.wildcards rhs.wildcards in
  let shared_wildcards = IntIntSet.inter lhs.wildcards rhs.wildcards in
  let wildcards_lhs = IntIntSet.filter (fun (r, _) -> IntSet.mem r (IntSet.diff rhs.rows lhs.rows)) all_wildcards in
  let wildcards_rhs = IntIntSet.filter (fun (r, _) -> IntSet.mem r (IntSet.diff rhs.rows lhs.rows)) all_wildcards in
  let new_preserved = IntIntSet.diff (IntIntSet.filter (fun (r, _) -> IntSet.mem r (IntSet.inter rhs.rows lhs.rows)) all_wildcards) shared_wildcards in
  {
    rows = IntSet.union lhs.rows rhs.rows;
    wildcards = IntIntSet.union wildcards_lhs (IntIntSet.union shared_wildcards wildcards_rhs);
    preserved_literals = IntIntSet.union new_preserved (IntIntSet.union lhs.preserved_literals rhs.preserved_literals);
    redundant = IntSet.inter lhs.redundant rhs.redundant;
  }

let get_wildcard_patterns (cinfo : complete_info) = IntIntSet.elements cinfo.wildcards |> List.map snd 
let get_preserved_patterns (cinfo : complete_info) = IntIntSet.elements cinfo.preserved_literals |> List.map snd |> IntSet.of_list

type 'a completeness =
  | Incomplete of 'a
  | Complete of complete_info
  | Completeness_unknown

let mk_complete ?redundant:(redundant = []) rows wildcards =
  Complete {
      rows = IntSet.of_list rows;
      wildcards = IntIntSet.of_list wildcards;
      preserved_literals = IntIntSet.empty;
      redundant = IntSet.of_list redundant
    }
 
let completeness_map f g = function
  | Incomplete exp -> Incomplete (f exp)
  | Complete cinfo -> Complete (g cinfo)
  | Completeness_unknown -> Completeness_unknown

(* turn a [t pat] into a [(t, int) pat] where each subpattern is uniquely identified *)
let number_pat (from : int) (pat : 'a pat) : ('a * int) pat * int =
  let rec go counter (P_aux (aux, (l, t))) =
    let count () = let c = !counter in (counter := c + 1; c) in
    let aux = match aux with
      | P_or (p1, p2) -> P_or (go counter p1, go counter p2)
      | P_not p -> P_not (go counter p)
      | P_as (p, id) -> P_as (go counter p, id)
      | P_typ (typ, p) -> P_typ (typ, go counter p)
      | P_var (p, tpat) -> P_var (go counter p, tpat)
      | P_app (ctor, ps) -> P_app (ctor, List.map (go counter) ps)
      | P_vector ps -> P_vector (List.map (go counter) ps)
      | P_vector_concat ps -> P_vector_concat (List.map (go counter) ps)
      | P_tup ps -> P_tup (List.map (go counter) ps)
      | P_list ps -> P_list (List.map (go counter) ps)
      | P_cons (p1, p2) -> P_cons (go counter p1, go counter p2)
      | P_string_append ps -> P_string_append (List.map (go counter) ps)
      | P_id id -> P_id id
      | P_lit lit -> P_lit lit
      | P_wild -> P_wild
    in
    P_aux (aux, (l, (t, count ())))
  in
  let counter = ref from in
  let pat = go counter pat in
  pat, !counter

let preserved_explanation =
  "Sail cannot simplify the above pattern match:\n"
  ^ "This bitvector pattern literal must be kept, as it is required for Sail to show that the surrounding pattern match is complete.\n"
  ^ "When translated into prover targets (e.g. Lem, Coq) without native bitvector patterns, they may be unable to verify completeness"
  
let rows_to_list (Rows rs) = rs
let columns_to_list (Columns cs) = cs 
 
type 'a rc_matrix = 'a columns rows
type 'a cr_matrix = 'a rows columns

let pop_column (matrix : 'a rc_matrix) : ((row_index * 'a) list * 'a rc_matrix) option =
  match rows_to_list matrix with
  | ((l, Columns (_ :: _)) :: _) as matrix ->
     Some (List.map (fun (l, row) -> (l, List.hd (columns_to_list row))) matrix, Rows (List.map (fun (l, row) -> (l, Columns (List.tl (columns_to_list row)))) matrix))
  | _ ->
     None
    
let rec transpose (matrix : 'a rc_matrix) : 'a cr_matrix =
  match pop_column matrix with
  | Some (col, matrix) -> Columns (Rows col :: columns_to_list (transpose matrix))
  | None -> Columns []

module Make(C: Config) = struct
  
  type bv_constraint =
    | BVC_eq of bv_constraint * bv_constraint
    | BVC_and of bv_constraint * bv_constraint
    | BVC_bvand of bv_constraint * bv_constraint
    | BVC_extract of int * int * bv_constraint
    | BVC_true
    | BVC_lit of string

  let rec string_of_bv_constraint = function
    | BVC_eq (bvc1, bvc2) -> "(= " ^ string_of_bv_constraint bvc1 ^ " " ^ string_of_bv_constraint bvc2 ^ ")"
    | BVC_and (bvc1, bvc2) -> "(and " ^ string_of_bv_constraint bvc1 ^ " " ^ string_of_bv_constraint bvc2 ^ ")"
    | BVC_bvand (bvc1, bvc2) -> "(bvand " ^ string_of_bv_constraint bvc1 ^ " " ^ string_of_bv_constraint bvc2 ^ ")"
    | BVC_extract (n, m, bvc) -> "((_ extract " ^ string_of_int n ^ " " ^ string_of_int m ^ ") " ^ string_of_bv_constraint bvc ^ ")"
    | BVC_true -> "true"
    | BVC_lit lit -> lit

  let bvc_and x y = match (x, y) with
    | BVC_true, _ -> y
    | _, BVC_true -> x
    | _, _ -> BVC_and (x, y)

  let typ_of_pat (P_aux (_, (_, (t, _)))) = C.typ_of_t t
  
  let insert_wildcards (cinfo : complete_info) (pat : (C.t * int) pat) : C.t pat =
    let preserved = get_preserved_patterns cinfo in
    let wildcards = get_wildcard_patterns cinfo in
    let rec go wild (P_aux (aux, (l, (t, n))) as full_pat) =
      if IntSet.mem n preserved then (
        Reporting.warn "Required literal" l preserved_explanation
      );
      let wild = wild || List.exists (fun wildcard -> wildcard = n) wildcards in
      let aux = match aux with
        | P_or (p1, p2) -> P_or (go wild p1, go wild p2)
        | P_not p -> P_not (go wild p)
        | P_as (p, id) -> P_as (go wild p, id)
        | P_typ (typ, p) -> P_typ (typ, go wild p)
        | P_var (p, tpat) -> P_var (go wild p, tpat)
        | P_app (ctor, ps) -> P_app (ctor, List.map (go wild) ps)
        | P_vector ps -> P_vector (List.map (go wild) ps)
        | P_vector_concat ps -> P_vector_concat (List.map (go wild) ps)
        | P_tup ps -> P_tup (List.map (go wild) ps)
        | P_list ps -> P_list (List.map (go wild) ps)
        | P_cons (p1, p2) -> P_cons (go wild p1, go wild p2)
        | P_string_append ps -> P_string_append (List.map (go wild) ps)
        | P_id id -> P_id id
        | P_lit _ when wild -> P_typ (typ_of_pat full_pat, P_aux (P_wild, (l, t)))
        | P_lit lit -> P_lit lit
        | P_wild -> P_wild
      in
      P_aux (aux, (l, t))
    in
    go false pat
 
  type gpat =
    | GP_wild
    | GP_unknown
    | GP_lit of lit
    | GP_tup of gpat list
    | GP_app of id * id * gpat list
    | GP_bitvector of int * int * (bv_constraint -> bv_constraint)
    | GP_enum of id * id
    | GP_vector of gpat list
    | GP_bool of bool

  let rec _string_of_gpat = function
    | GP_wild -> "_"
    | GP_unknown -> "?"
    | GP_lit lit -> string_of_lit lit
    | GP_tup gpats -> "(" ^ Util.string_of_list ", " _string_of_gpat gpats ^ ")"
    | GP_app (_, ctor, gpats) ->
       string_of_id ctor ^ "(" ^ Util.string_of_list ", " _string_of_gpat gpats ^ ")"
    | GP_bitvector (_, _, bvc) -> string_of_bv_constraint (bvc (BVC_lit "x"))
    | GP_enum (_, id) -> string_of_id id
    | GP_bool b -> string_of_bool b
    | GP_vector gpats -> "[" ^ Util.string_of_list ", " _string_of_gpat gpats ^ "]"
 
  let rec generalize ctx (P_aux (p_aux, (l, (_, pnum))) as pat) =
    let typ = typ_of_pat pat in
    match p_aux with
    | P_lit (L_aux (L_unit, _)) ->
       (* Unit pattern always matches on unit, so generalize to wildcard *)
       GP_wild
      
    | P_lit (L_aux (L_hex hex, _)) -> GP_bitvector (pnum, String.length hex * 4, fun x -> BVC_eq (x, BVC_lit ("#x" ^ hex)))
    | P_lit (L_aux (L_bin bin, _)) -> GP_bitvector (pnum, String.length bin, fun x -> BVC_eq (x, BVC_lit ("#b" ^ bin)))
                                    
    | P_vector pats when is_bitvector_typ typ ->
       let mask, bits =
         List.fold_left (fun (mask, bits) (P_aux (pat, _)) ->
             let rec go pat = match pat with
               | P_lit (L_aux (L_one, _)) -> (mask ^ "1", bits ^ "1")
               | P_lit (L_aux (L_zero, _)) -> (mask ^ "1", bits ^ "0")
               | P_wild | P_id _ -> (mask ^ "0", bits ^ "0")
               | P_typ (_, P_aux (pat, _)) -> go pat 
               | _ ->
                  Reporting.warn "Unexpected pattern" l "";
                  (mask ^ "0", bits ^ "0")
             in
             go pat
           ) ("#b", "#b") pats
       in
       GP_bitvector (pnum, List.length pats, fun x -> BVC_eq (BVC_bvand (BVC_lit mask, x), BVC_lit bits))

    | P_vector pats ->
       GP_vector (List.map (generalize ctx) pats)
       
    | P_vector_concat pats when is_bitvector_typ typ ->
       let lengths =
         List.fold_left (fun acc typ ->
             match acc with
             | None -> None
             | Some lengths ->
                let (nexp, _, _) = vector_typ_args_of typ in
                match int_of_nexp_opt nexp with
                | Some n -> Some (Big_int.to_int n :: lengths)
                | None -> None
           ) (Some []) (List.map typ_of_pat pats) in
       let gpats = List.map (generalize ctx) pats in
       begin match lengths with
       | Some lengths ->
          let (total, slices) = List.fold_left (fun (total, acc) len -> (total + len, (total + len - 1, total) :: acc)) (0, []) lengths in
          let bvc = fun x ->
            List.fold_left2 (fun bvc (n, m) gpat ->
                match gpat with
                | GP_bitvector (_, _, bvc_subpat) ->
                   bvc_and bvc (bvc_subpat (BVC_extract (n, m, x)))
                | GP_wild ->
                   bvc
                | _ ->
                   Reporting.unreachable l __POS__ "Invalid bitvector pattern"
              ) BVC_true slices gpats
          in
          GP_bitvector (pnum, total, bvc)
       | None ->
          GP_wild
       end

    | P_tup pats ->
       GP_tup (List.map (generalize ctx) pats)
       
    | P_app (id, pats) ->
       let typ_id = match typ with
         | Typ_aux (Typ_app (id, _), _) -> id
         | Typ_aux (Typ_id id, _) -> id
         | _ -> failwith "Bad type"
       in
       GP_app (typ_id, id, List.map (generalize ctx) pats)

    | P_lit (L_aux (L_true, _)) -> GP_bool true
    | P_lit (L_aux (L_false, _)) -> GP_bool false
    | P_lit lit -> GP_lit lit
    | P_wild -> GP_wild
    | P_var (pat, _) -> generalize ctx pat
    | P_as (pat, _) -> generalize ctx pat
    | P_typ (_, pat) -> generalize ctx pat

    | P_id id ->
       begin match List.find_opt (fun (enum, ctors) -> IdSet.mem id ctors) (Bindings.bindings ctx.enums) with
       | Some (enum, _) -> GP_enum (enum, id)
       | None -> GP_wild
       end

    | _ -> GP_unknown

  let rec find_smtlib_type = function
    | (_, GP_bitvector (_, len, _)) :: _ -> Some ("(_ BitVec " ^ string_of_int len ^ ")")
    | _ :: rest -> find_smtlib_type rest
    | [] -> None

  let is_simple_gpat = function
    | GP_bitvector _ -> true
    | GP_wild -> true
    | _ -> false

  let rec column_type = function
    | (_, GP_tup gpats) :: _ -> Tuple_column (List.length gpats)
    | (_, GP_app (typ_id, _, _)) :: _ -> App_column typ_id
    | (_, GP_bool _) :: _ -> Bool_column
    | (_, GP_enum (typ_id, _)) :: _ -> Enum_column typ_id
    | (_, GP_lit _) :: _ -> Lit_column
    | _ :: rest -> column_type rest
    | [] -> Unknown_column

  let rec unmatched_string_literal max_length = function
    | (_, GP_lit (L_aux (L_string str, _))) :: rest -> unmatched_string_literal (max (String.length str) max_length) rest
    | _ :: rest -> unmatched_string_literal max_length rest
    | [] -> L_string (String.make (max_length + 1) '?')

  let rec unmatched_num_literal n = function
    | (_, GP_lit (L_aux (L_num m, _))) :: rest -> unmatched_num_literal (Big_int.max n m) rest
    | _ :: rest -> unmatched_num_literal n rest
    | [] -> L_num (Big_int.succ n)
          
  let rec unmatched_literal = function
    | (_, GP_lit (L_aux (L_string str, _))) :: rest -> Some (unmatched_string_literal (String.length str) rest)
    | (_, GP_lit (L_aux (L_num n, _))) :: rest -> Some (unmatched_num_literal n rest)
    | _ :: rest -> unmatched_literal rest
    | [] -> None
 
  let simple_matrix_is_complete _ctx matrix =
    let vars =
      List.mapi (fun i (Rows column) ->
          match find_smtlib_type column with
          | None -> None
          | Some ty -> Some (i, ty)
        ) (columns_to_list (transpose matrix))
    in
    let just_vars = vars |> Util.option_these in
    let all_rows = List.map (fun (idx, _) -> idx.num) (rows_to_list matrix) in
    match just_vars with
    | [] -> mk_complete all_rows [] (* The entire matrix is wildcard patterns *)
    | _ -> 
       let constrs =
         List.map (fun (l, Columns row) ->
             let row_constrs =
               List.map2 (fun var gpat ->
                   match var, gpat with
                   | (Some (i, _), GP_bitvector (_, _, bvc)) -> Some (string_of_bv_constraint (bvc (BVC_lit ("v" ^ string_of_int i))))
                   | _ -> None
                 ) vars row
               |> Util.option_these
             in
             match row_constrs with
             | [] -> (l, None)
             | _ -> (l, Some ("(assert (not (and " ^ Util.string_of_list " " (fun x -> x) row_constrs ^ ")))"))
           ) (rows_to_list matrix)
       in
       (* Check if we have a row containing only wildcards, hence matrix is trivially unsatisfiable *)
       match Util.find_rest_opt (fun (_, constr) -> Util.is_none constr) constrs with
       | Some (_, []) -> mk_complete all_rows []
       (* If there are any rows after the wildcard row, they are redundant *)
       | Some (_, redundant) ->
          mk_complete ~redundant:(List.map (fun (idx, _) -> idx.num) redundant) [] []
       | None ->
          let smtlib =
            Util.string_of_list "\n" (fun (v, ty) -> Printf.sprintf "(declare-const v%d %s)" v ty) just_vars ^ "\n"
            ^ Util.string_of_list "\n" (fun x -> x) (Util.option_these (List.map snd constrs)) ^ "\n"
            ^ "(check-sat)\n"
            ^ "(get-model)\n"
          in
          match Constraint.call_smt_solve_bitvector Parse_ast.Unknown smtlib just_vars with
          | Some lits ->
             Incomplete (List.init (List.length vars) (fun i -> match List.assoc_opt i lits with
                                                                | Some lit -> mk_exp (E_lit lit)
                                                                | None -> mk_lit_exp L_undef))
          | None ->
             let to_wildcards = match Util.last_opt (rows_to_list matrix) with
               | Some (idx, Columns row) -> Util.map_filter (function GP_bitvector (pnum, _, _) -> Some (idx.num, pnum) | _ -> None) row
               | None -> []
             in
             mk_complete all_rows to_wildcards
         
  let find_complex_column matrix =
    let is_complex_column col = List.exists (fun (_, gpat) -> not (is_simple_gpat gpat)) col in
    let columns = List.mapi (fun i col -> (i, rows_to_list col)) (columns_to_list (transpose matrix)) in
    List.find_opt (fun (_, col) -> is_complex_column col) columns

  let rec column_typ_id l = function
    | (_, GP_app (typ_id, _, _)) :: _ -> typ_id
    | _ :: gpats -> column_typ_id l gpats
    | [] -> Reporting.unreachable l __POS__ "No column type id"
                  
  let split_app_column l ctx col =
    let typ_id = column_typ_id l col in
    let all_ctors = Bindings.find typ_id ctx.variants |> snd |> List.map (function Tu_aux (Tu_ty_id (_, id), _) -> id) in
    let all_ctors = List.fold_left (fun m ctor -> Bindings.add ctor [] m) Bindings.empty all_ctors in
    List.fold_left (fun (i, acc) (_, gpat) ->
        let acc = match gpat with
          | GP_app (_, ctor, ctor_gpats) ->
             Bindings.update ctor (function None -> Some [(i, Some ctor_gpats)] | Some xs -> Some ((i, Some ctor_gpats) :: xs)) acc
          | GP_wild ->
             Bindings.map (fun xs -> (i, None) :: xs) acc
          | _ ->
             Reporting.unreachable Parse_ast.Unknown __POS__ "App column contains invalid pattern"
        in
        (i + 1, acc)
      ) (0, all_ctors) col
    |> snd

  let flatten_tuple_column width i matrix =
    let flatten = function
      | GP_tup gpats -> gpats
      | GP_wild -> List.init width (fun _ -> GP_wild)
      | _ -> Reporting.unreachable Parse_ast.Unknown __POS__ "Tuple column contains invalid pattern"
    in
    Rows (List.map (fun (l, row) ->
              (l, Columns (List.mapi (fun j gpat -> if i = j then flatten gpat else [gpat]) (columns_to_list row) |> List.concat))
            ) (rows_to_list matrix))
 
  let split_matrix_ctor ctx c ctor ctor_rows matrix =
    let row_indices = List.fold_left (fun set (r, _) -> IntSet.add r set) IntSet.empty ctor_rows in
    let flatten = function
      | GP_app (_, _, gpats) -> GP_tup gpats
      | GP_wild -> GP_wild
      | _ -> Reporting.unreachable Parse_ast.Unknown __POS__ "App column contains invalid pattern"
    in
    let remove_ctor row = Columns (List.mapi (fun i gpat -> if i = c then flatten gpat else gpat) (columns_to_list row)) in
    Rows (
        rows_to_list matrix
        |> List.mapi (fun r row -> (r, row))
        |> Util.map_filter (fun (r, (l, row)) -> if IntSet.mem r row_indices then Some (l, remove_ctor row) else None)
      )

  let rec remove_index n = function
    | x :: xs when n = 0 -> xs
    | x :: xs -> x :: remove_index (n - 1) xs
    | [] -> []
    
  let split_matrix_bool b c matrix =
    let is_bool_row = function
      | GP_bool b' -> b = b'
      | GP_wild -> true
      | _ -> false
    in
    Rows (
        rows_to_list matrix
        |> List.filter (fun (_, row) -> columns_to_list row |> (fun xs -> List.nth xs c) |> is_bool_row)
        |> List.map (fun (l, row) -> (l, Columns (remove_index c (columns_to_list row))))
      )

  let split_matrix_wild c matrix =
    let is_wild_row = function
      | GP_wild -> true
      | _ -> false
    in
    Rows (
        rows_to_list matrix
        |> List.filter (fun (_, row) -> columns_to_list row |> (fun xs -> List.nth xs c) |> is_wild_row)
        |> List.map (fun (l, row) -> (l, Columns (remove_index c (columns_to_list row))))
      )
 
  let split_matrix_enum e c matrix =
    let is_enum_row = function
      | GP_enum (_, id) -> Id.compare e id = 0
      | GP_wild -> true
      | _ -> false
    in
    Rows (
        rows_to_list matrix
        |> List.filter (fun (_, row) -> columns_to_list row |> (fun xs -> List.nth xs c) |> is_enum_row)
        |> List.map (fun (l, row) -> (l, Columns (remove_index c (columns_to_list row))))
      )
    
  let retuple width i unmatcheds =
    let (xs, ys) = Util.split_after i unmatcheds in
    let tuple_elems = Util.take width ys in
    let zs = Util.drop width ys in
    xs @ mk_exp (E_tuple tuple_elems) :: zs

  let rector ctor i unmatcheds =
    let (xs, ys) = Util.split_after i unmatcheds in
    match ys with
    | E_aux (E_tuple args, _) :: zs ->
       xs @ mk_exp (E_app (ctor, args)) :: zs
    | y :: zs ->
       xs @ mk_exp (E_app (ctor, [y])) :: zs
    | [] ->
       xs @ [mk_exp (E_app (ctor, []))]

  let relit lit i unmatcheds =
    let (xs, ys) = Util.split_after i unmatcheds in
    xs @ mk_lit_exp lit :: ys
      
  let rebool b i unmatcheds =
    relit (if b then L_true else L_false) i unmatcheds

  let reenum e i unmatcheds =
    let (xs, ys) = Util.split_after i unmatcheds in
    xs @ mk_exp (E_id e) :: ys
    
  let row_matrix_empty (Rows rows) =
    match rows with
    | [] -> true
    | _ -> false

  let row_matrix_width l (Rows rows) =
    match rows with
    | (_, Columns cols) :: _ -> List.length cols
    | [] -> Reporting.unreachable l __POS__ "Cannot determine width of empty pattern matrix"

  let rec undefs_except n c v len =
    if n = len then
      []
    else if n = c then
      v :: undefs_except (n + 1) c v len
    else
      mk_lit_exp L_undef :: undefs_except (n + 1) c v len
 
  let rec matrix_is_complete l ctx matrix =
    match find_complex_column matrix with
    | None -> simple_matrix_is_complete ctx matrix
    | Some (i, col) ->
       begin match column_type col with
       | Tuple_column width ->
          matrix_is_complete l ctx (flatten_tuple_column width i matrix)
          |> completeness_map (retuple width i) (fun w -> w)

       | Lit_column ->
          let wild_matrix = split_matrix_wild i matrix in
          begin match unmatched_literal col with
          | None -> Completeness_unknown
          | Some lit ->
             if row_matrix_empty wild_matrix then
               Incomplete (undefs_except 0 i (mk_lit_exp lit) (row_matrix_width l matrix))
             else
               match matrix_is_complete l ctx wild_matrix with
               | Incomplete unmatcheds -> Incomplete (relit lit i unmatcheds)
               | Complete cinfo -> Complete cinfo
               | Completeness_unknown -> Completeness_unknown
          end
            
       | App_column typ_id ->
          let ctors = split_app_column l ctx col in
          Bindings.fold (fun ctor ctor_rows unmatcheds ->
              match unmatcheds with
              | Incomplete unmatcheds -> Incomplete unmatcheds
              | Completeness_unknown -> Completeness_unknown
              | Complete cinfo ->
                 let ctor_matrix = split_matrix_ctor ctx i ctor ctor_rows matrix in
                 if row_matrix_empty ctor_matrix then
                   let width = row_matrix_width l matrix in
                   Incomplete (undefs_except 0 i (mk_exp (E_app (ctor, [mk_lit_exp L_undef]))) width)
                 else
                   matrix_is_complete l ctx ctor_matrix |> completeness_map (rector ctor i) (union_complete cinfo)
            ) ctors (mk_complete [] [])

       | Bool_column ->
          let true_matrix = split_matrix_bool true i matrix in
          let false_matrix = split_matrix_bool false i matrix in
          let width = row_matrix_width l matrix in
          if row_matrix_empty true_matrix then
            Incomplete (undefs_except 0 i (mk_lit_exp L_true) width)
          else if row_matrix_empty false_matrix then
            Incomplete (undefs_except 0 i (mk_lit_exp L_false) width)
          else
            begin match matrix_is_complete l ctx true_matrix with
            | Incomplete unmatcheds ->
               Incomplete (rebool true i unmatcheds)
            | Complete cinfo ->
               matrix_is_complete l ctx false_matrix |> completeness_map (rebool false i) (union_complete cinfo)
            | Completeness_unknown ->
               Completeness_unknown
            end

       | Enum_column typ_id ->
          let members = Bindings.find typ_id ctx.enums in
          IdSet.fold (fun member unmatcheds ->
              match unmatcheds with
              | Incomplete unmatcheds -> Incomplete unmatcheds
              | Completeness_unknown -> Completeness_unknown
              | Complete cinfo ->
                 let enum_matrix = split_matrix_enum member i matrix in
                 if row_matrix_empty enum_matrix then
                   let width = row_matrix_width l matrix in
                   Incomplete (undefs_except 0 i (mk_exp (E_id member)) width)
                 else
                   matrix_is_complete l ctx enum_matrix |> completeness_map (reenum member i) (union_complete cinfo)
            ) members (mk_complete [] [])
 
       | Unknown_column -> Completeness_unknown
       end
 
  (* Just highlight the match keyword and not the whole match block. *)
  let shrink_loc = function
    | Parse_ast.Range (n, m) ->
       Lexing.(Parse_ast.Range (n, { n with pos_cnum = n.pos_cnum + 5 }))
    | l -> l

  let rec cases_to_pats from have_guard = function
    | [] -> have_guard, []
    | Pat_aux (Pat_exp ((P_aux (_, (l, _)) as pat), _), _) :: cases ->
       let pat, from = number_pat from pat in
       let have_guard, pats = cases_to_pats from have_guard cases in
       have_guard, ((l, pat) :: pats)
    (* We don't consider guarded cases *)
    | Pat_aux (Pat_when _, _) :: cases -> cases_to_pats from true cases

  let rec update_cases l new_pats cases =
    match new_pats, cases with
    | [], [] -> []
    | (new_pat :: new_pats), (Pat_aux (Pat_exp (_, exp), annot) :: cases) ->
       Pat_aux (Pat_exp (new_pat, exp), annot) :: update_cases l new_pats cases
    | _, ((Pat_aux (Pat_when _, _) as case) :: cases) ->
       case :: update_cases l new_pats cases
    | _, _ ->
       Reporting.unreachable l __POS__ "Impossible case in update_cases"
 
  let is_complete_wildcarded l ctx cases typ =
    try
      match cases_to_pats 0 false cases with
      | _, [] -> None
      | have_guard, pats ->
         let matrix = Rows (List.mapi (fun i (l, pat) -> ({ loc = l; num = i}, Columns [generalize ctx pat])) pats) in
         begin match matrix_is_complete l ctx matrix with
         | Incomplete (unmatched :: _) ->
            let guard_info = if have_guard then " by unguarded patterns" else "" in
            Reporting.warn "Incomplete pattern match statement at" (shrink_loc l)
              ("The following expression is unmatched" ^ guard_info ^ ": " ^ (string_of_exp unmatched |> Util.yellow |> Util.clear));
            None
         | Incomplete [] ->
            Reporting.unreachable l __POS__ "Got unmatched pattern matrix without witness"
         | Complete cinfo ->
            let wildcarded_pats = List.map (fun (_, pat) -> insert_wildcards cinfo pat) pats in
            List.iter (fun (idx, _) ->
                if IntSet.mem idx.num cinfo.redundant then
                  Reporting.warn "Redundant case" idx.loc "This match case is never used"
              ) (rows_to_list matrix);
            Some (update_cases l wildcarded_pats cases)
         | Completeness_unknown ->
            None
         end
    with
    (* For now, if any error occurs just report the pattern match is incomplete *)
    | exn -> None

  let is_complete l ctx cases typ = Util.is_some (is_complete_wildcarded l ctx cases typ)
           
end
