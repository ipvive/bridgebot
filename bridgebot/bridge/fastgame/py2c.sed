s/    \( *\)/\1\1/;
s/  *$//;
s/else:.*/} else {/;
s/elif \(.*\):.*/} else if (\1) {/;
s/if \(.*\):.*/if (\1) {/;
s/ and / \&\& /g;
s/self\.\([A-Z_][A-Z_]*\)/\1/;
s/self\./state->/g;
s/ = None/ = NA/;
s/\(.*\) = \(.*\)/\1 = \2;/;
s/def _*\(.*\)(self, *\(.*\)):/static TYPE \1(GameState *state, TYPES \2) {/;
s/return *$/return;/;

# these last two work in vim but not in sed.
s/\[\([^,\]]*\), *\([^,\]]*\)\]/[\1][\2]/
s/\[\([^,\]]*\), *\([^,\]]*\), *\([^,\]]*\)\]/[\1][\2]/
