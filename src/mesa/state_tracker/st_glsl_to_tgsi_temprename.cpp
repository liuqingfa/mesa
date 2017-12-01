/*
 * Copyright © 2017 Gert Wollny
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "st_glsl_to_tgsi_temprename.h"
#include "st_glsl_to_tgsi_array_merge.h"

#include <tgsi/tgsi_info.h>
#include <tgsi/tgsi_strings.h>
#include <program/prog_instruction.h>
#include <limits>
#include <cstdlib>

/* std::sort is significantly faster than qsort */
#define USE_STL_SORT
#ifdef USE_STL_SORT
#include <algorithm>
#endif

#ifndef NDEBUG
#include <iostream>
#include <iomanip>
#include <program/prog_print.h>
#include <util/debug.h>
using std::cerr;
using std::setw;
#endif

/* If <windows.h> is included this is defined and clashes with
 * std::numeric_limits<>::max()
 */
#ifdef max
#undef max
#endif

using std::numeric_limits;

/* Without c++11 define the nullptr for forward-compatibility
 * and better readibility */
#if __cplusplus < 201103L
#define nullptr 0
#endif

#ifndef NDEBUG
/* Helper function to check whether we want to seen debugging output */
static inline bool is_debug_enabled ()
{
   static int debug_enabled = -1;
   if (debug_enabled < 0)
      debug_enabled = env_var_as_boolean("GLSL_TO_TGSI_RENAME_DEBUG", false);
   return debug_enabled > 0;
}
#define RENAME_DEBUG(X) if (is_debug_enabled()) do { X; } while (false);
#else
#define RENAME_DEBUG(X)
#endif

namespace {

enum prog_scope_type {
   outer_scope,           /* Outer program scope */
   loop_body,             /* Inside a loop */
   if_branch,             /* Inside if branch */
   else_branch,           /* Inside else branch */
   switch_body,           /* Inside switch statmenet */
   switch_case_branch,    /* Inside switch case statmenet */
   switch_default_branch, /* Inside switch default statmenet */
   undefined_scope
};

class prog_scope {
public:
   prog_scope(prog_scope *parent, prog_scope_type type, int id,
              int depth, int begin);

   prog_scope_type type() const;
   prog_scope *parent() const;
   int nesting_depth() const;
   int id() const;
   int end() const;
   int begin() const;
   int loop_break_line() const;

   const prog_scope *in_else_scope() const;
   const prog_scope *in_ifelse_scope() const;
   const prog_scope *in_parent_ifelse_scope() const;
   const prog_scope *innermost_loop() const;
   const prog_scope *outermost_loop() const;
   const prog_scope *enclosing_conditional() const;

   bool is_loop() const;
   bool is_in_loop() const;
   bool is_switchcase_scope_in_loop() const;
   bool is_conditional() const;
   bool is_child_of(const prog_scope *scope) const;
   bool is_child_of_ifelse_id_sibling(const prog_scope *scope) const;

   bool break_is_for_switchcase() const;
   bool contains_range_of(const prog_scope& other) const;

   void set_end(int end);
   void set_loop_break_line(int line);

private:
   prog_scope_type scope_type;
   int scope_id;
   int scope_nesting_depth;
   int scope_begin;
   int scope_end;
   int break_loop_line;
   prog_scope *parent_scope;
};

/* Some storage class to encapsulate the prog_scope (de-)allocations */
class prog_scope_storage {
public:
   prog_scope_storage(void *mem_ctx, int n);
   ~prog_scope_storage();
   prog_scope * create(prog_scope *p, prog_scope_type type, int id,
                       int lvl, int s_begin);
private:
   void *mem_ctx;
   int current_slot;
   prog_scope *storage;
};

/* Class to track the access to a component of a temporary register. */

class temp_comp_access {
public:
   temp_comp_access();

   void record_read(int line, prog_scope *scope);
   void record_write(int line, prog_scope *scope);
   register_lifetime get_required_lifetime();
private:
   void propagate_lifetime_to_dominant_write_scope();
   bool conditional_ifelse_write_in_loop() const;

   void record_ifelse_write(const prog_scope& scope);
   void record_if_write(const prog_scope& scope);
   void record_else_write(const prog_scope& scope);

   prog_scope *last_read_scope;
   prog_scope *first_read_scope;
   prog_scope *first_write_scope;

   int first_write;
   int last_read;
   int last_write;
   int first_read;

   /* This member variable tracks the current resolution of conditional writing
    * to this temporary in IF/ELSE clauses.
    *
    * The initial value "conditionality_untouched" indicates that this
    * temporary has not yet been written to within an if clause.
    *
    * A positive (other than "conditionality_untouched") number refers to the
    * last loop id for which the write was resolved as unconditional. With each
    * new loop this value will be overwitten by "conditionality_unresolved"
    * on entering the first IF clause writing this temporary.
    *
    * The value "conditionality_unresolved" indicates that no resolution has
    * been achieved so far. If the variable is set to this value at the end of
    * the processing of the whole shader it also indicates a conditional write.
    *
    * The value "write_is_conditional" marks that the variable is written
    * conditionally (i.e. not in all relevant IF/ELSE code path pairs) in at
    * least one loop.
    */
   int conditionality_in_loop_id;

   /* Helper constants to make the tracking code more readable. */
   static const int write_is_conditional = -1;
   static const int conditionality_unresolved = 0;
   static const int conditionality_untouched;

   /* A bit field tracking the nexting levels of if-else clauses where the
    * temporary has (so far) been written to in the if branch, but not in the
    * else branch.
    */
   unsigned int if_scope_write_flags;

   int next_ifelse_nesting_depth;
   static const int supported_ifelse_nesting_depth = 32;

   /* Tracks the last if scope in which the temporary was written to
    * without a write in the correspondig else branch. Is also used
    * to track read-before-write in the according scope.
    */
   const prog_scope *current_unpaired_if_write_scope;

   /* Flag to resolve read-before-write in the else scope. */
   bool was_written_in_current_else_scope;
};

const int
temp_comp_access::conditionality_untouched = numeric_limits<int>::max();

/* Class to track the access to all components of a temporary register. */
class temp_access {
public:
   temp_access();
   void record_read(int line, prog_scope *scope, int swizzle);
   void record_write(int line, prog_scope *scope, int writemask);
   register_lifetime get_required_lifetime();
private:
   void update_access_mask(int mask);

   temp_comp_access comp[4];
   int access_mask;
   bool needs_component_tracking;
};

class array_access {
public:
   array_access();
   void record_read(int line, prog_scope *scope, int swizzle);
   void record_write(int line, prog_scope *scope, int writemask);
   void get_required_lifetime(array_lifetime &lt);
private:
   int first_access;
   int last_access;
   prog_scope *first_access_scope;
   prog_scope *last_access_scope;
   bool conditional_write_in_loop;
   int accumulated_swizzle;
};

prog_scope_storage::prog_scope_storage(void *mc, int n):
   mem_ctx(mc),
   current_slot(0)
{
   storage = ralloc_array(mem_ctx, prog_scope, n);
}

prog_scope_storage::~prog_scope_storage()
{
   //ralloc_free(storage);
}

prog_scope*
prog_scope_storage::create(prog_scope *p, prog_scope_type type, int id,
                           int lvl, int s_begin)
{
   storage[current_slot] = prog_scope(p, type, id, lvl, s_begin);
   return &storage[current_slot++];
}

prog_scope::prog_scope(prog_scope *parent, prog_scope_type type, int id,
                       int depth, int scope_begin):
   scope_type(type),
   scope_id(id),
   scope_nesting_depth(depth),
   scope_begin(scope_begin),
   scope_end(-1),
   break_loop_line(numeric_limits<int>::max()),
   parent_scope(parent)
{
}

prog_scope_type prog_scope::type() const
{
   return scope_type;
}

prog_scope *prog_scope::parent() const
{
   return parent_scope;
}

int prog_scope::nesting_depth() const
{
   return scope_nesting_depth;
}

bool prog_scope::is_loop() const
{
   return (scope_type == loop_body);
}

bool prog_scope::is_in_loop() const
{
   if (scope_type == loop_body)
      return true;

   if (parent_scope)
      return parent_scope->is_in_loop();

   return false;
}

const prog_scope *prog_scope::innermost_loop() const
{
   if (scope_type == loop_body)
      return this;

   if (parent_scope)
      return parent_scope->innermost_loop();

   return nullptr;
}

const prog_scope *prog_scope::outermost_loop() const
{
   const prog_scope *loop = nullptr;
   const prog_scope *p = this;

   do {
      if (p->type() == loop_body)
         loop = p;
      p = p->parent();
   } while (p);

   return loop;
}

bool prog_scope::is_child_of_ifelse_id_sibling(const prog_scope *scope) const
{
   const prog_scope *my_parent = in_parent_ifelse_scope();
   while (my_parent) {
      /* is a direct child? */
      if (my_parent == scope)
         return false;
      /* is a child of the conditions sibling? */
      if (my_parent->id() == scope->id())
         return true;
      my_parent = my_parent->in_parent_ifelse_scope();
   }
   return false;
}

bool prog_scope::is_child_of(const prog_scope *scope) const
{
   const prog_scope *my_parent = parent();
   while (my_parent) {
      if (my_parent == scope)
         return true;
      my_parent = my_parent->parent();
   }
   return false;
}

const prog_scope *prog_scope::enclosing_conditional() const
{
   if (is_conditional())
      return this;

   if (parent_scope)
      return parent_scope->enclosing_conditional();

   return nullptr;
}

bool prog_scope::contains_range_of(const prog_scope& other) const
{
   return (begin() <= other.begin()) && (end() >= other.end());
}

bool prog_scope::is_conditional() const
{
   return scope_type == if_branch ||
         scope_type == else_branch ||
         scope_type == switch_case_branch ||
         scope_type == switch_default_branch;
}

const prog_scope *prog_scope::in_else_scope() const
{
   if (scope_type == else_branch)
      return this;

   if (parent_scope)
      return parent_scope->in_else_scope();

   return nullptr;
}

const prog_scope *prog_scope::in_parent_ifelse_scope() const
{
   if (parent_scope)
      return parent_scope->in_ifelse_scope();
   else
      return nullptr;
}

const prog_scope *prog_scope::in_ifelse_scope() const
{
   if (scope_type == if_branch ||
       scope_type == else_branch)
      return this;

   if (parent_scope)
      return parent_scope->in_ifelse_scope();

   return nullptr;
}

bool prog_scope::is_switchcase_scope_in_loop() const
{
   return (scope_type == switch_case_branch ||
           scope_type == switch_default_branch) &&
         is_in_loop();
}

bool prog_scope::break_is_for_switchcase() const
{
   if (scope_type == loop_body)
      return false;

   if (scope_type == switch_case_branch ||
       scope_type == switch_default_branch ||
       scope_type == switch_body)
      return true;

   if (parent_scope)
      return parent_scope->break_is_for_switchcase();

   return false;
}

int prog_scope::id() const
{
   return scope_id;
}

int prog_scope::begin() const
{
   return scope_begin;
}

int prog_scope::end() const
{
   return scope_end;
}

void prog_scope::set_end(int end)
{
   if (scope_end == -1)
      scope_end = end;
}

void prog_scope::set_loop_break_line(int line)
{
   if (scope_type == loop_body) {
      break_loop_line = MIN2(break_loop_line, line);
   } else {
      if (parent_scope)
         parent()->set_loop_break_line(line);
   }
}

int prog_scope::loop_break_line() const
{
   return break_loop_line;
}

temp_access::temp_access():
   access_mask(0),
   needs_component_tracking(false)
{
}

void temp_access::update_access_mask(int mask)
{
   if (access_mask && access_mask != mask)
      needs_component_tracking = true;
   access_mask |= mask;
}

void temp_access::record_write(int line, prog_scope *scope, int writemask)
{
   update_access_mask(writemask);

   if (writemask & WRITEMASK_X)
      comp[0].record_write(line, scope);
   if (writemask & WRITEMASK_Y)
      comp[1].record_write(line, scope);
   if (writemask & WRITEMASK_Z)
      comp[2].record_write(line, scope);
   if (writemask & WRITEMASK_W)
      comp[3].record_write(line, scope);
}

void temp_access::record_read(int line, prog_scope *scope, int readmask)
{
   update_access_mask(readmask);

   if (readmask & WRITEMASK_X)
      comp[0].record_read(line, scope);
   if (readmask & WRITEMASK_Y)
      comp[1].record_read(line, scope);
   if (readmask & WRITEMASK_Z)
      comp[2].record_read(line, scope);
   if (readmask & WRITEMASK_W)
      comp[3].record_read(line, scope);
}

inline static register_lifetime make_lifetime(int b, int e)
{
   register_lifetime lt;
   lt.begin = b;
   lt.end = e;
   return lt;
}

register_lifetime temp_access::get_required_lifetime()
{
   register_lifetime result = make_lifetime(-1, -1);

   unsigned mask = access_mask;
   while (mask) {
      unsigned chan = u_bit_scan(&mask);
      register_lifetime lt = comp[chan].get_required_lifetime();

      if (lt.begin >= 0) {
         if ((result.begin < 0) || (result.begin > lt.begin))
            result.begin = lt.begin;
      }

      if (lt.end > result.end)
         result.end = lt.end;

      if (!needs_component_tracking)
         break;
   }
   return result;
}

temp_comp_access::temp_comp_access():
   last_read_scope(nullptr),
   first_read_scope(nullptr),
   first_write_scope(nullptr),
   first_write(-1),
   last_read(-1),
   last_write(-1),
   first_read(numeric_limits<int>::max()),
   conditionality_in_loop_id(conditionality_untouched),
   if_scope_write_flags(0),
   next_ifelse_nesting_depth(0),
   current_unpaired_if_write_scope(nullptr),
   was_written_in_current_else_scope(false)
{
}

void temp_comp_access::record_read(int line, prog_scope *scope)
{
   last_read_scope = scope;
   last_read = line;

   if (first_read > line) {
      first_read = line;
      first_read_scope = scope;
   }

   /* Check whether we are in a condition within a loop */
   const prog_scope *ifelse_scope = scope->in_ifelse_scope();
   const prog_scope *enclosing_loop;
   if (ifelse_scope && (enclosing_loop = ifelse_scope->innermost_loop())) {

      /* If we have either not yet written to this register nor writes are
       * resolved as unconditional in the enclosing loop then check whether
       * we read before write in an IF/ELSE branch.
       */
      if ((conditionality_in_loop_id != write_is_conditional) &&
          (conditionality_in_loop_id != enclosing_loop->id())) {

         if (current_unpaired_if_write_scope)  {

            /* Has been written in this or a parent scope? - this makes the temporary
             * unconditionally set at this point.
             */
            if (scope->is_child_of(current_unpaired_if_write_scope))
               return;

            /* Has been written in the same scope before it was read? */
            if (ifelse_scope->type() == if_branch) {
               if (current_unpaired_if_write_scope->id() == scope->id())
                  return;
            } else {
               if (was_written_in_current_else_scope)
                  return;
            }
         }

         /* The temporary was read (conditionally) before it is written, hence
          * it should survive a loop. This can be signaled like if it were
          * conditionally written.
          */
         conditionality_in_loop_id = write_is_conditional;
      }
   }
}

void temp_comp_access::record_write(int line, prog_scope *scope)
{
   last_write = line;

   if (first_write < 0) {
      first_write = line;
      first_write_scope = scope;
   }

   if (conditionality_in_loop_id == write_is_conditional)
      return;

   /* If the nesting depth is larger than the supported level,
    * then we assume conditional writes.
    */
   if (next_ifelse_nesting_depth >= supported_ifelse_nesting_depth) {
      conditionality_in_loop_id = write_is_conditional;
      return;
   }

   /* If we are in an IF/ELSE scope within a loop and the loop has not
    * been resolved already, then record this write.
    */
   const prog_scope *ifelse_scope = scope->in_ifelse_scope();
   if (ifelse_scope && ifelse_scope->innermost_loop() &&
       ifelse_scope->innermost_loop()->id()  != conditionality_in_loop_id)
      record_ifelse_write(*ifelse_scope);
}

void temp_comp_access::record_ifelse_write(const prog_scope& scope)
{
   if (scope.type() == if_branch) {
      /* The first write in an IF branch within a loop implies unresolved
       * conditionality (if it was untouched or unconditional before).
       */
      conditionality_in_loop_id = conditionality_unresolved;
      was_written_in_current_else_scope = false;
      record_if_write(scope);
   } else {
      was_written_in_current_else_scope = true;
      record_else_write(scope);
   }
}

void temp_comp_access::record_if_write(const prog_scope& scope)
{
   /* Don't record write if this IF scope if it ...
    * - is not the first write in this IF scope,
    * - has already been written in a parent IF scope.
    * In both cases this write is a secondary write that doesn't contribute
    * to resolve conditionality.
    *
    * Record the write if it
    * - is the first one (obviously),
    * - happens in an IF branch that is a child of the ELSE branch of the
    *   last active IF/ELSE pair. In this case recording this write is used to
    *   established whether the write is (un-)conditional in the scope enclosing
    *   this outer IF/ELSE pair.
    */
   if (!current_unpaired_if_write_scope ||
       (current_unpaired_if_write_scope->id() != scope.id() &&
        scope.is_child_of_ifelse_id_sibling(current_unpaired_if_write_scope)))  {
      if_scope_write_flags |= 1 << next_ifelse_nesting_depth;
      current_unpaired_if_write_scope = &scope;
      next_ifelse_nesting_depth++;
   }
}

void temp_comp_access::record_else_write(const prog_scope& scope)
{
   int mask = 1 << (next_ifelse_nesting_depth - 1);

   /* If the temporary was written in an IF branch on the same scope level
    * and this branch is the sibling of this ELSE branch, then we have a
    * pair of writes that makes write access to this temporary unconditional
    * in the enclosing scope.
    */

   if ((if_scope_write_flags & mask) &&
       (scope.id() == current_unpaired_if_write_scope->id())) {
          --next_ifelse_nesting_depth;
         if_scope_write_flags &= ~mask;

         /* The following code deals with propagating unconditionality from
          * inner levels of nested IF/ELSE to the outer levels like in
          *
          * 1: var t;
          * 2: if (a) {        <- start scope A
          * 3:    if (b)
          * 4:         t = ...
          * 5:    else
          * 6:         t = ...
          * 7: } else {        <- start scope B
          * 8:    if (c)
          * 9:         t = ...
          * A:    else         <- start scope C
          * B:         t = ...
          * C: }
          *
          */

         const prog_scope *parent_ifelse = scope.parent()->in_ifelse_scope();

         if (1 << (next_ifelse_nesting_depth - 1) & if_scope_write_flags) {
            /* We are at the end of scope C and already recorded a write
             * within an IF scope (A), the sibling of the parent ELSE scope B,
             * and it is not yet resolved. Mark that as the last relevant
             * IF scope. Below the write will be resolved for the A/B
             * scope pair.
             */
            current_unpaired_if_write_scope = parent_ifelse;
         } else {
            current_unpaired_if_write_scope = nullptr;
         }

         /* If some parent is IF/ELSE and in a loop then propagate the
          * write to that scope. Otherwise the write is unconditional
          * because it happens in both corresponding IF/ELSE branches
          * in this loop, and hence, record the loop id to signal the
          * resolution.
          */
         if (parent_ifelse && parent_ifelse->is_in_loop()) {
            record_ifelse_write(*parent_ifelse);
         } else {
            conditionality_in_loop_id = scope.innermost_loop()->id();
         }
   } else {
     /* The temporary was not written in the IF branch corresponding
      * to this ELSE branch, hence the write is conditional.
      */
      conditionality_in_loop_id = write_is_conditional;
   }
}

bool temp_comp_access::conditional_ifelse_write_in_loop() const
{
   return conditionality_in_loop_id <= conditionality_unresolved;
}

void temp_comp_access::propagate_lifetime_to_dominant_write_scope()
{
   first_write = first_write_scope->begin();
   int lr = first_write_scope->end();

   if (last_read < lr)
      last_read = lr;
}

register_lifetime temp_comp_access::get_required_lifetime()
{
   bool keep_for_full_loop = false;

   /* This register component is not used at all, or only read,
    * mark it as unused and ignore it when renaming.
    * glsl_to_tgsi_visitor::renumber_registers will take care of
    * eliminating registers that are not written to.
    */
   if (last_write < 0)
      return make_lifetime(-1, -1);

   assert(first_write_scope);

   /* Only written to, just make sure the register component is not
    * reused in the range it is used to write to
    */
   if (!last_read_scope)
      return make_lifetime(first_write, last_write + 1);

   const prog_scope *enclosing_scope_first_read = first_read_scope;
   const prog_scope *enclosing_scope_first_write = first_write_scope;

   /* We read before writing in a loop
    * hence the value must survive the loops
    */
   if ((first_read <= first_write) &&
       first_read_scope->is_in_loop()) {
      keep_for_full_loop = true;
      enclosing_scope_first_read = first_read_scope->outermost_loop();
   }

   /* A conditional write within a (nested) loop must survive the outermost
    * loop if the last read was not within the same scope.
    */
   const prog_scope *conditional = enclosing_scope_first_write->enclosing_conditional();
   if (conditional && !conditional->contains_range_of(*last_read_scope) &&
       (conditional->is_switchcase_scope_in_loop() ||
        conditional_ifelse_write_in_loop())) {
         keep_for_full_loop = true;
         enclosing_scope_first_write = conditional->outermost_loop();
   }

   /* Evaluate the scope that is shared by all: required first write scope,
    * required first read before write scope, and last read scope.
    */
   const prog_scope *enclosing_scope = enclosing_scope_first_read;
   if (enclosing_scope_first_write->contains_range_of(*enclosing_scope))
      enclosing_scope = enclosing_scope_first_write;

   if (last_read_scope->contains_range_of(*enclosing_scope))
      enclosing_scope = last_read_scope;

   while (!enclosing_scope->contains_range_of(*enclosing_scope_first_write) ||
          !enclosing_scope->contains_range_of(*last_read_scope)) {
      enclosing_scope = enclosing_scope->parent();
      assert(enclosing_scope);
   }

   /* Propagate the last read scope to the target scope */
   while (enclosing_scope->nesting_depth() < last_read_scope->nesting_depth()) {
      /* If the read is in a loop and we have to move up the scope we need to
       * extend the life time to the end of this current loop because at this
       * point we don't know whether the component was written before
       * un-conditionally in the same loop.
       */
      if (last_read_scope->is_loop())
         last_read = last_read_scope->end();

      last_read_scope = last_read_scope->parent();
   }

   /* If the variable has to be kept for the whole loop, and we
    * are currently in a loop, then propagate the life time.
    */
   if (keep_for_full_loop && first_write_scope->is_loop())
      propagate_lifetime_to_dominant_write_scope();

   /* Propagate the first_dominant_write scope to the target scope */
   while (enclosing_scope->nesting_depth() < first_write_scope->nesting_depth()) {
      /* Propagate lifetime if there was a break in a loop and the write was
       * after the break inside that loop. Note, that this is only needed if
       * we move up in the scopes.
       */
      if (first_write_scope->loop_break_line() < first_write) {
         keep_for_full_loop = true;
         propagate_lifetime_to_dominant_write_scope();
      }

      first_write_scope = first_write_scope->parent();

      /* Propagte lifetime if we are now in a loop */
      if (keep_for_full_loop && first_write_scope->is_loop())
          propagate_lifetime_to_dominant_write_scope();
   }

   /* The last write past the last read is dead code, but we have to
    * ensure that the component is not reused too early, hence extend the
    * lifetime past the last write.
    */
   if (last_write >= last_read)
      last_read = last_write + 1;

   /* Here we are at the same scope, all is resolved */
   return make_lifetime(first_write, last_read);
}

array_access::array_access():
   first_access(-1),
   last_access(-1),
   first_access_scope(nullptr),
   last_access_scope(nullptr),
   conditional_write_in_loop(false),
   accumulated_swizzle(0)
{
}

void array_access::record_read(int line, prog_scope *scope, int swizzle)
{
   if (!first_access_scope) {
      first_access = line;
      first_access_scope = scope;
   }
   last_access_scope = scope;
   last_access = line;
   accumulated_swizzle |= swizzle;
}

void array_access::record_write(int line, prog_scope *scope, int writemask)
{
   if (!first_access_scope) {
      first_access = line;
      first_access_scope = scope;
   }
   last_access_scope = scope;
   last_access = line;
   accumulated_swizzle |= writemask;
   if (scope->in_ifelse_scope() && scope->innermost_loop())
      conditional_write_in_loop = true;
}

void array_access::get_required_lifetime(array_lifetime& lt)
{
   if (first_access_scope == last_access_scope) {
      lt.set_lifetime(first_access, last_access);
      lt.set_access_mask(accumulated_swizzle);
   }

   const prog_scope *shared_scope = first_access_scope;
   const prog_scope *other_scope = last_access_scope;

   assert(shared_scope);
   RENAME_DEBUG(cerr << "shared_scope=" << shared_scope << "\n");

   if (conditional_write_in_loop) {
      const prog_scope *help = shared_scope->outermost_loop();
      if (help) {
         shared_scope = help;
      } else {
         help = other_scope->outermost_loop();
         if (help)
            other_scope = help;
      }
      if (first_access > shared_scope->begin())
         first_access = shared_scope->begin();
      if (last_access < shared_scope->end())
         last_access = shared_scope->end();
   }

   /* See if any of the two is the parent of the other. */
   if (other_scope->contains_range_of(*shared_scope)) {
      shared_scope = other_scope;
   } else while (!shared_scope->contains_range_of(*other_scope)) {
      assert(shared_scope->parent());
      if (shared_scope->type() == loop_body) {
         if (last_access < shared_scope->end())
             last_access = shared_scope->end();
      }
      shared_scope = shared_scope->parent();
   }

   while (shared_scope != other_scope) {
      if (other_scope->type() == loop_body) {
         if (last_access < other_scope->end())
             last_access = other_scope->end();
      }
      other_scope = other_scope->parent();
   }

   lt.set_lifetime(first_access, last_access);
   lt.set_access_mask(accumulated_swizzle);
}

/* Helper class for sorting and searching the registers based
 * on life times. */
class temp_access_record {
public:
   int begin;
   int end;
   int reg;
   bool erase;

   bool operator < (const temp_access_record& rhs) const {
      return begin < rhs.begin;
   }
};

class access_recorder {
public:
   access_recorder(int _ntemps, int _narrays);
   ~access_recorder();

   void record_read(const st_src_reg& src, int line, prog_scope *scope);
   void record_write(const st_dst_reg& src, int line, prog_scope *scope);

   void get_required_lifetimes(struct register_lifetime *reg_lifetimes,
                               struct array_lifetime *arr_lifetimes);
private:

   int ntemps;
   int narrays;
   temp_access *acc;
   array_access *arr;
};

access_recorder::access_recorder(int _ntemps, int _narrays):
   ntemps(_ntemps),
   narrays(_narrays)
{
   acc = new temp_access[ntemps];
   arr = new array_access[narrays];
}

access_recorder::~access_recorder()
{
   delete[] arr;
   delete[] acc;
}

void access_recorder::record_read(const st_src_reg& src, int line,
                                  prog_scope *scope)
{
   int readmask = 0;
   for (int idx = 0; idx < 4; ++idx) {
      int swz = GET_SWZ(src.swizzle, idx);
      readmask |= (1 << swz) & 0xF;
   }

   if (src.file == PROGRAM_TEMPORARY)
      acc[src.index].record_read(line, scope, readmask);

   if (src.file == PROGRAM_ARRAY) {
      RENAME_DEBUG(cerr << "src.array_id=" << src.array_id << ", narray="
                   << narrays  << " read scope: " << scope << "\n");
      assert(src.array_id <= narrays);
      arr[src.array_id - 1].record_read(line, scope, readmask);
   }

   if (src.reladdr)
      record_read(*src.reladdr, line, scope);
   if (src.reladdr2)
      record_read(*src.reladdr2, line, scope);
}

void access_recorder::record_write(const st_dst_reg& dst, int line,
                                   prog_scope *scope)
{
   if (dst.file == PROGRAM_TEMPORARY)
      acc[dst.index].record_write(line, scope, dst.writemask);

   if (dst.file == PROGRAM_ARRAY) {
      RENAME_DEBUG(cerr << "dst.array_id=" << dst.array_id << ", narray="
                   << narrays << " write scope: " << scope << "\n");
      assert(dst.array_id <= narrays);
      arr[dst.array_id - 1].record_write(line, scope, dst.writemask);
   }

   if (dst.reladdr)
      record_read(*dst.reladdr, line, scope);
   if (dst.reladdr2)
      record_read(*dst.reladdr2, line, scope);
}

void access_recorder::get_required_lifetimes(struct register_lifetime *reg_lifetimes,
                                             struct array_lifetime *arr_lifetimes)
{
   RENAME_DEBUG(cerr << "========= register lifetimes ==============\n");
   for(int i = 0; i < ntemps; ++i) {
      RENAME_DEBUG(cerr << setw(4) << i);
      reg_lifetimes[i] = acc[i].get_required_lifetime();
      RENAME_DEBUG(cerr << ": [" << reg_lifetimes[i].begin<< ", "
                   << reg_lifetimes[i].end << "]\n");
   }
   RENAME_DEBUG(cerr << "==================================\n\n");

   RENAME_DEBUG(cerr << "========= array lifetimes ("<< narrays
                << ")==============" << std::endl);
   for(int i = 0; i < narrays; ++i) {
      RENAME_DEBUG(cerr << setw(4) << i);
      arr[i].get_required_lifetime(arr_lifetimes[i]);
      RENAME_DEBUG(cerr << ": [" << arr_lifetimes[i].begin() << ", "
                   << arr_lifetimes[i].end() << "]\n");
   }
   RENAME_DEBUG(cerr << "==================================\n\n");
}

}
#ifndef NDEBUG
/* Function used for debugging. */
static void dump_instruction(std::ostream& os, int line, prog_scope *scope,
                             const glsl_to_tgsi_instruction& inst);
#endif

/* Scan the program and estimate the required register life times.
 * The array lifetimes must be pre-allocated
 */
bool
get_temp_registers_required_lifetimes(void *mem_ctx, exec_list *instructions,
                                      int ntemps, struct register_lifetime *reg_lifetimes,
                                      int narrays,
                                      array_lifetime *arr_lifetimes)
{
   int line = 0;
   int loop_id = 1;
   int if_id = 1;
   int switch_id = 0;
   bool is_at_end = false;
   int n_scopes = 1;

   /* Count scopes to allocate the needed space without the need for
    * re-allocation
    */
   foreach_in_list(glsl_to_tgsi_instruction, inst, instructions) {
      if (inst->op == TGSI_OPCODE_BGNLOOP ||
          inst->op == TGSI_OPCODE_SWITCH ||
          inst->op == TGSI_OPCODE_CASE ||
          inst->op == TGSI_OPCODE_IF ||
          inst->op == TGSI_OPCODE_UIF ||
          inst->op == TGSI_OPCODE_ELSE ||
          inst->op == TGSI_OPCODE_DEFAULT)
         ++n_scopes;
   }

   prog_scope_storage scopes(mem_ctx, n_scopes);

   access_recorder access(ntemps, narrays);

   prog_scope *cur_scope = scopes.create(nullptr, outer_scope, 0, 0, line);

   RENAME_DEBUG(cerr << "========= Begin shader ============\n");

   foreach_in_list(glsl_to_tgsi_instruction, inst, instructions) {
      if (is_at_end) {
         assert(!"GLSL_TO_TGSI: shader has instructions past end marker");
         break;
      }

      RENAME_DEBUG(dump_instruction(cerr, line, cur_scope, *inst));

      switch (inst->op) {
      case TGSI_OPCODE_BGNLOOP: {
         cur_scope = scopes.create(cur_scope, loop_body, loop_id++,
                                   cur_scope->nesting_depth() + 1, line);
         break;
      }
      case TGSI_OPCODE_ENDLOOP: {
         cur_scope->set_end(line);
         cur_scope = cur_scope->parent();
         assert(cur_scope);
         break;
      }
      case TGSI_OPCODE_IF:
      case TGSI_OPCODE_UIF: {
         assert(num_inst_src_regs(inst) == 1);
         access.record_read(inst->src[0], line, cur_scope);
         cur_scope = scopes.create(cur_scope, if_branch, if_id++,
                                   cur_scope->nesting_depth() + 1, line + 1);
         break;
      }
      case TGSI_OPCODE_ELSE: {
         assert(cur_scope->type() == if_branch);
         cur_scope->set_end(line - 1);
         cur_scope = scopes.create(cur_scope->parent(), else_branch,
                                   cur_scope->id(), cur_scope->nesting_depth(),
                                   line + 1);
         break;
      }
      case TGSI_OPCODE_END: {
         cur_scope->set_end(line);
         is_at_end = true;
         break;
      }
      case TGSI_OPCODE_ENDIF: {
         cur_scope->set_end(line - 1);
         cur_scope = cur_scope->parent();
         assert(cur_scope);
         break;
      }
      case TGSI_OPCODE_SWITCH: {
         assert(num_inst_src_regs(inst) == 1);
         prog_scope *scope = scopes.create(cur_scope, switch_body, switch_id++,
                                           cur_scope->nesting_depth() + 1, line);
         /* We record the read only for the SWITCH statement itself, like it
          * is used by the only consumer of TGSI_OPCODE_SWITCH in tgsi_exec.c.
          */
         access.record_read(inst->src[0], line, cur_scope);
         cur_scope = scope;
         break;
      }
      case TGSI_OPCODE_ENDSWITCH: {
         cur_scope->set_end(line - 1);
         /* Remove the case level, it might not have been
          * closed with a break.
          */
         if (cur_scope->type() != switch_body)
            cur_scope = cur_scope->parent();

         cur_scope = cur_scope->parent();
         assert(cur_scope);
         break;
      }
      case TGSI_OPCODE_CASE: {
         /* Take care of tracking the registers. */
         prog_scope *switch_scope = cur_scope->type() == switch_body ?
                                       cur_scope : cur_scope->parent();

         assert(num_inst_src_regs(inst) == 1);
         access.record_read(inst->src[0], line, switch_scope);

         /* Fall through to allocate the scope. */
      }
      case TGSI_OPCODE_DEFAULT: {
         prog_scope_type t = inst->op == TGSI_OPCODE_CASE ? switch_case_branch
                                                          : switch_default_branch;
         prog_scope *switch_scope = (cur_scope->type() == switch_body) ?
                                       cur_scope : cur_scope->parent();
         assert(switch_scope->type() == switch_body);
         prog_scope *scope = scopes.create(switch_scope, t,
                                           switch_scope->id(),
                                           switch_scope->nesting_depth() + 1,
                                           line);
         /* Previous case falls through, so scope was not yet closed. */
         if ((cur_scope != switch_scope) && (cur_scope->end() == -1))
            cur_scope->set_end(line - 1);
         cur_scope = scope;
         break;
      }
      case TGSI_OPCODE_BRK: {
         if (cur_scope->break_is_for_switchcase()) {
            cur_scope->set_end(line - 1);
         } else {
            cur_scope->set_loop_break_line(line);
         }
         break;
      }
      case TGSI_OPCODE_CAL:
      case TGSI_OPCODE_RET:
         /* These opcodes are not supported and if a subroutine would
          * be called in a shader, then the lifetime tracking would have
          * to follow that call to see which registers are used there.
          * Since this is not done, we have to bail out here and signal
          * that no register merge will take place.
          */
         return false;
      default: {
         for (unsigned j = 0; j < num_inst_src_regs(inst); j++) {
            access.record_read(inst->src[j], line, cur_scope);
         }
         for (unsigned j = 0; j < inst->tex_offset_num_offset; j++) {
            access.record_read(inst->tex_offsets[j], line, cur_scope);
         }
         for (unsigned j = 0; j < num_inst_dst_regs(inst); j++) {
            access.record_write(inst->dst[j], line, cur_scope);
         }
      }
      }
      ++line;
   }


   RENAME_DEBUG(cerr << "==================================\n\n");

   /* Make sure last scope is closed, even though no
    * TGSI_OPCODE_END was given.
    */
   if (cur_scope->end() < 0)
      cur_scope->set_end(line - 1);

   access.get_required_lifetimes(reg_lifetimes, arr_lifetimes);
   return true;
}

/* Find the next register between [start, end) that has a life time starting
 * at or after bound by using a binary search.
 * start points at the beginning of the search range,
 * end points at the element past the end of the search range, and
 * the array comprising [start, end) must be sorted in ascending order.
 */
template <typename T>
static T* find_next_rename(T* start, T* end, int bound)
{
   int delta = (end - start);

   while (delta > 0) {
      int half = delta >> 1;
      T* middle = start + half;

      if (bound <= middle->begin) {
         delta = half;
      } else {
         start = middle;
         ++start;
         delta -= half + 1;
      }
   }

   return start;
}

#ifndef USE_STL_SORT
static int access_record_compare (const void *a, const void *b) {
   const access_record *aa = static_cast<const access_record*>(a);
   const access_record *bb = static_cast<const access_record*>(b);
   return aa->begin < bb->begin ? -1 : (aa->begin > bb->begin ? 1 : 0);
}
#endif

/* This functions evaluates the register merges by using a binary
 * search to find suitable merge candidates. */
void get_temp_registers_remapping(void *mem_ctx, int ntemps,
                                  const struct register_lifetime* lifetimes,
                                  struct rename_reg_pair *result)
{
   temp_access_record *reg_access = ralloc_array(mem_ctx, temp_access_record, ntemps);

   int used_temps = 0;
   for (int i = 0; i < ntemps; ++i) {
      if (lifetimes[i].begin >= 0) {
         reg_access[used_temps].begin = lifetimes[i].begin;
         reg_access[used_temps].end = lifetimes[i].end;
         reg_access[used_temps].reg = i;
         reg_access[used_temps].erase = false;
         ++used_temps;
      }
   }

#ifdef USE_STL_SORT
   std::sort(reg_access, reg_access + used_temps);
#else
   std::qsort(reg_access, used_temps, sizeof(access_record), access_record_compare);
#endif

   temp_access_record *trgt = reg_access;
   temp_access_record *reg_access_end = reg_access + used_temps;
   temp_access_record *first_erase = reg_access_end;
   temp_access_record *search_start = trgt + 1;

   while (trgt != reg_access_end) {
      temp_access_record *src = find_next_rename(search_start, reg_access_end,
                                                 trgt->end);
      if (src != reg_access_end) {
         result[src->reg].new_reg = trgt->reg;
         result[src->reg].valid = true;
         trgt->end = src->end;

         /* Since we only search forward, don't remove the renamed
          * register just now, only mark it. */
         src->erase = true;

         if (first_erase == reg_access_end)
            first_erase = src;

         search_start = src + 1;
      } else {
         /* Moving to the next target register it is time to remove
          * the already merged registers from the search range */
         if (first_erase != reg_access_end) {
            temp_access_record *outp = first_erase;
            temp_access_record *inp = first_erase + 1;

            while (inp != reg_access_end) {
               if (!inp->erase)
                  *outp++ = *inp;
               ++inp;
            }

            reg_access_end = outp;
            first_erase = reg_access_end;
         }
         ++trgt;
         search_start = trgt + 1;
      }
   }
   ralloc_free(reg_access);
}

/* Code below used for debugging */
#ifndef NDEBUG
static const char swizzle_txt[] = "xyzw";

static const char *tgsi_file_names[PROGRAM_FILE_MAX] =  {
   "TEMP",  "ARRAY",   "IN", "OUT", "STATE", "CONST",
   "UNIFORM",  "WO", "ADDR", "SAMPLER",  "SV", "UNDEF",
   "IMM", "BUF",  "MEM",  "IMAGE"
};

template <typename st_reg>
void dump_reg_access(std::ostream& os, const st_reg& reg)
{
   os << tgsi_file_names[reg.file];
   if (reg.file == PROGRAM_ARRAY)
      os << "(" << reg.array_id << ")";

   if (reg.has_index2) {
      os << "[";
      if (reg.reladdr2)
         os << *reg.reladdr2 << "+";
      os << reg.index2D << "]";
   }

   os << "[";
   if (reg.reladdr)
      os << *reg.reladdr << "+";
   os << reg.index << "]";
}

static std::ostream& operator << (std::ostream& os, const st_src_reg& reg)
{
   dump_reg_access(os, reg);

   if (reg.swizzle != SWIZZLE_XYZW) {
      os << ".";
      for (int idx = 0; idx < 4; ++idx) {
         int swz = GET_SWZ(reg.swizzle, idx);
         if (swz < 4) {
            os << swizzle_txt[swz];
         }
      }
   }
   return os;
}

static std::ostream& operator << (std::ostream& os, const st_dst_reg& dst)
{
   dump_reg_access(os, dst);

   if (dst.writemask != TGSI_WRITEMASK_XYZW) {
      os << ".";
      if (dst.writemask & TGSI_WRITEMASK_X) os << "x";
      if (dst.writemask & TGSI_WRITEMASK_Y) os << "y";
      if (dst.writemask & TGSI_WRITEMASK_Z) os << "z";
      if (dst.writemask & TGSI_WRITEMASK_W) os << "w";
   }

   return os;
}

static
void dump_instruction(std::ostream& os, int line, prog_scope *scope,
                      const glsl_to_tgsi_instruction& inst)
{
   const struct tgsi_opcode_info *info = tgsi_get_opcode_info(inst.op);

   int indent = scope->nesting_depth();
   if ((scope->type() == switch_case_branch ||
        scope->type() == switch_default_branch) &&
       (info->opcode == TGSI_OPCODE_CASE ||
        info->opcode == TGSI_OPCODE_DEFAULT))
      --indent;

   if (info->opcode == TGSI_OPCODE_ENDIF ||
       info->opcode == TGSI_OPCODE_ELSE ||
       info->opcode == TGSI_OPCODE_ENDLOOP ||
       info->opcode == TGSI_OPCODE_ENDSWITCH)
      --indent;

   os << setw(4) << line << ": ";
   for (int i = 0; i < indent; ++i)
      os << "    ";
   os << tgsi_get_opcode_name(info->opcode) << " ";

   bool has_operators = false;
   for (unsigned j = 0; j < num_inst_dst_regs(&inst); j++) {
      has_operators = true;
      if (j > 0)
         os << ", ";

      os << inst.dst[j];

   }
   if (has_operators)
      os << " := ";

   for (unsigned j = 0; j < num_inst_src_regs(&inst); j++) {
      if (j > 0)
         os << ", ";
      os << inst.src[j];
   }

   if (inst.tex_offset_num_offset > 0) {
      os << ", TEXOFS: ";
      for (unsigned j = 0; j < inst.tex_offset_num_offset; j++) {
         if (j > 0)
            os << ", ";
         os << inst.tex_offsets[j];
      }
   }
   os << "\n";
}
#endif
