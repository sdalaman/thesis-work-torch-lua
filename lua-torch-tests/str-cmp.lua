

function strcmp(str1,str2)
  local l1 = str1:len()
  local l2 = str2:len()
  cnt = 0
  for i=1,l1 do
    if i > l2 then
        break
    end
    if str1:sub(i,i) == str2:sub(i,i) then
      cnt = cnt + 1
    else
      break
    end
  end
  return cnt,cnt/l1
end


s1 = "sabandalaman"
s2 = "saban"
s3 = "sabandal"
s4 = "dal"

c,r = strcmp(s1,s2)
print("s2 "..c.." "..r)

c,r = strcmp(s1,s3)
print("s3 "..c.." "..r)

c,r = strcmp(s1,s4)
print("s4 "..c.." "..r)

print("end")